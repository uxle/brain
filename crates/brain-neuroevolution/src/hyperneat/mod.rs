//! # HyperNEAT (Hypercube NeuroEvolution of Augmenting Topologies)
//!
//! Substrate geometry, geometric query points, and Compositional Pattern-Producing Networks (CPPN).
#![allow(missing_docs)]

pub mod cppn;
pub mod substrate;

pub use cppn::{Cppn, CppnNode, CppnActivation};
pub use substrate::{SubstrateGrid2D, SubstrateConfig};

/// Configuration for HyperNEAT.
#[derive(Debug, Clone, Default)]
pub struct HyperneatConfig {
    pub max_nodes: usize,
    pub weight_threshold: f64,
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
