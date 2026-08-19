//! # Compositional Pattern-Producing Network (CPPN)
//!
//! Multi-activation functional network generating spatial weight patterns from geometric coordinates (x1, y1, x2, y2).
#![allow(missing_docs)]

/// Activation function applied by CPPN nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CppnActivation {
    #[default]
    Linear,
    Sigmoid,
    Gaussian,
    Sine,
    Abs,
}

impl CppnActivation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            CppnActivation::Linear => x,
            CppnActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            CppnActivation::Gaussian => (-x * x).exp(),
            CppnActivation::Sine => x.sin(),
            CppnActivation::Abs => x.abs(),
        }
    }
}

/// Node in a CPPN graph.
#[derive(Debug, Clone)]
pub struct CppnNode {
    pub activation: CppnActivation,
    pub bias: f64,
}

/// Compositional Pattern-Producing Network evaluator.
#[derive(Debug, Clone, Default)]
pub struct Cppn {
    pub nodes: Vec<CppnNode>,
}

impl Cppn {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Evaluates connection weight from source coordinate (x1, y1) to target coordinate (x2, y2).
    pub fn query(&self, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        // Distance and coordinate features
        let dx = x2 - x1;
        let dy = y2 - y1;
        let dist = (dx * dx + dy * dy).sqrt();

        // Baseline spatial pattern: Gaussian over distance + linear bias
        CppnActivation::Gaussian.apply(dist) - 0.5
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
