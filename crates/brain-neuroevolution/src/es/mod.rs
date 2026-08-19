//! # Evolution Strategies (ES)
//!
//! (1+1)-ES with 1/5th success rule and Covariance Matrix Adaptation (CMA-ES).
#![allow(missing_docs)]

pub mod cmaes;
pub mod es1p1;

pub use cmaes::{Cmaes, CmaesConfig};
pub use es1p1::{Es1p1, Es1p1Config};

/// Evolution Strategy algorithm kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EsKind {
    #[default]
    CMAES,
    Es1p1,
}

/// Result returned from Evolution Strategy optimization.
#[derive(Debug, Clone, Default)]
pub struct EsResult {
    pub best_params: Vec<f64>,
    pub best_fitness: f64,
    pub evaluations: usize,
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
