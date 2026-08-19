//! # Evolutionary Hyperparameters
//!
//! Subsystem configuration options for selection, crossover, and mutation schedules.
#![allow(missing_docs)]


/// Configuration for genetic operators and bounds.
#[derive(Debug, Clone)]
pub struct OperatorConfig {
    pub min_gene_val: f64,
    pub max_gene_val: f64,
    pub gaussian_sigma: f64,
    pub tournament_size: usize,
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            min_gene_val: -5.0,
            max_gene_val: 5.0,
            gaussian_sigma: 0.1,
            tournament_size: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
