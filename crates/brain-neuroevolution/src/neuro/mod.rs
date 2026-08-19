//! # Neuroevolution Parameter Mapping
//!
//! Maps flat evolutionary genome vectors to structured neural network layers and weights.
#![allow(missing_docs)]

pub mod weights;
pub use weights::{LayerWeightDescriptor, flatten_layer_weights, unflatten_layer_weights};


/// Configuration for neuroevolution mapping.
#[derive(Debug, Clone, Default)]
pub struct NeuroConfig {
    pub layer_shapes: Vec<Vec<usize>>,
}

/// Computes the total number of scalar weight parameters required across all layer shapes.
pub fn total_neuro_parameters(shapes: &[Vec<usize>]) -> usize {
    shapes.iter().map(|s| s.iter().product::<usize>()).sum()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
}
