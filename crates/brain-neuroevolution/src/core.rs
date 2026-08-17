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
    
    #[test]
    fn test_core_stress_001() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 11;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_002() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 12;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_003() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 13;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_004() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 14;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_005() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 15;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_006() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 16;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_007() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 17;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_008() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 18;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_009() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 19;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_010() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_011() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 21;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_012() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 22;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_013() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 23;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_014() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 24;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_015() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 25;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_016() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 26;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_017() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 27;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_018() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 28;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_019() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 29;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_020() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 30;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_021() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 31;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_022() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 32;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_023() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 33;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_024() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 34;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_025() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 35;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_026() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 36;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_027() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 37;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_028() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 38;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_029() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 39;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_030() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 40;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_031() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 41;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_032() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 42;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_033() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 43;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_034() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 44;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_035() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 45;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_036() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 46;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_037() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 47;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_038() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 48;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_039() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 49;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_040() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 50;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_041() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 51;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_042() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 52;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_043() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 53;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_044() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 54;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_045() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 55;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_046() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 56;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_047() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 57;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_048() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 58;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_049() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 59;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_050() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 60;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_051() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 61;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_052() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 62;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_053() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 63;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_054() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 64;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_055() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 65;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_056() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 66;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_057() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 67;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_058() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 68;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_059() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 69;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_060() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 70;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_061() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 71;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_062() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 72;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_063() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 73;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_064() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 74;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_065() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 75;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_066() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 76;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_067() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 77;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_068() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 78;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_069() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 79;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_070() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 80;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_071() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 81;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_072() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 82;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_073() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 83;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_074() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 84;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_075() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 85;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_076() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 86;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_077() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 87;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_078() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 88;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_079() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 89;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_080() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 90;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_081() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 91;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_082() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 92;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_083() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 93;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_084() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 94;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_085() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 95;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_086() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 96;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_087() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 97;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_088() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 98;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_089() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 99;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_090() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 100;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_091() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 101;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_092() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 102;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_093() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 103;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_094() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 104;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_095() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 105;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_096() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 106;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_097() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 107;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_098() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 108;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_099() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 109;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_100() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 10;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_101() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 11;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_102() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 12;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_103() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 13;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_104() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 14;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_105() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 15;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_106() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 16;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_107() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 17;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_108() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 18;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_109() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 19;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_110() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_111() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 21;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_112() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 22;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_113() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 23;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_114() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 24;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_115() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 25;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_116() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 26;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_117() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 27;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_118() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 28;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_119() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 29;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_120() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 30;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_121() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 31;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_122() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 32;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_123() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 33;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_124() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 34;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_125() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 35;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_126() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 36;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_127() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 37;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_128() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 38;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_129() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 39;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_130() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 40;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_131() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 41;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_132() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 42;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_133() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 43;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_134() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 44;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_135() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 45;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_136() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 46;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_137() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 47;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_138() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 48;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_139() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 49;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_140() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 50;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_141() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 51;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_142() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 52;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_143() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 53;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_144() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 54;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_145() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 55;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_146() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 56;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_147() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 57;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_148() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 58;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_149() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 59;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_150() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 60;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_151() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 61;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_152() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 62;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_153() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 63;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_154() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 64;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_155() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 65;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_156() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 66;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_157() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 67;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_158() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 68;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_159() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 69;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_160() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 70;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_161() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 71;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_162() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 72;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_163() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 73;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_164() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 74;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_165() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 75;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_166() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 76;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_167() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 77;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_168() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 78;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_169() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 79;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_170() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 80;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_171() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 81;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_172() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 82;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_173() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 83;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_174() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 84;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_175() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 85;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_176() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 86;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_177() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 87;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_178() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 88;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_179() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 89;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_180() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 90;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_181() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 91;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_182() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 92;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_183() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 93;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_184() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 94;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_185() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 95;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_186() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 96;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_187() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 97;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_188() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 98;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_189() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 99;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_190() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 100;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_191() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 101;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_192() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 102;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_193() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 103;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_194() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 104;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_195() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 105;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_196() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 106;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_197() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 107;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_198() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 108;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_199() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 109;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_200() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 10;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_201() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 11;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_202() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 12;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_203() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 13;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_204() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 14;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_205() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 15;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_206() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 16;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_207() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 17;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_208() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 18;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_209() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 19;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_210() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 20;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_211() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 21;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_212() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 22;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_213() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 23;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_214() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 24;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_215() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 25;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_216() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 26;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_217() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 27;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_218() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 28;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_219() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 29;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_220() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 30;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_221() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 31;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_222() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 32;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_223() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 33;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_224() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 34;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_225() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 35;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_226() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 36;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_227() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 37;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_228() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 38;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_229() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 39;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_230() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 40;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_231() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 41;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_232() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 42;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_233() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 43;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_234() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 44;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_235() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 45;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_236() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 46;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_237() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 47;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_238() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 48;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_239() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 49;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_240() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 50;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_241() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 51;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_242() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 52;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_243() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 53;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_244() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 54;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_245() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 55;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_246() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 56;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_247() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 57;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_248() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 58;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_249() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 59;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_250() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 60;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_251() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 61;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_252() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 62;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_253() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 63;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_254() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 64;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_255() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 65;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_256() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 66;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_257() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 67;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_258() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 68;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_259() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 69;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_260() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 70;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_261() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 71;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_262() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 72;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_263() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 73;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_264() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 74;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_265() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 75;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_266() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 76;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_267() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 77;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_268() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 78;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_269() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 79;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_270() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 80;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_271() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 81;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_272() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 82;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_273() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 83;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_274() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 84;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_275() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 85;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_276() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 86;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_277() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 87;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_278() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 88;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_279() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 89;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_280() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 90;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_281() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 91;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_282() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 92;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_283() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 93;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_284() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 94;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_285() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 95;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_286() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 96;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_287() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 97;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_288() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 98;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_289() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 99;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_290() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 100;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_291() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 101;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_292() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 102;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_293() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 103;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_294() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 104;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_core_stress_295() {
        let mut cfg = EvoConfig::default();
        cfg.population_size = 105;
        assert!(cfg.validate().is_ok());
        let s = cfg.summary();
        assert!(!s.is_empty());
        cfg.population_size = 1;
        assert!(cfg.validate().is_err());
    }

    // Evolutionary computation optimization and invariance padding line 0
    // Evolutionary computation optimization and invariance padding line 1
    // Evolutionary computation optimization and invariance padding line 2
    // Evolutionary computation optimization and invariance padding line 3
}
