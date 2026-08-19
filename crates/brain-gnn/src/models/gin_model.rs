//! # GIN Model
//!
//! Multi-layer GIN model with batch-norm / MLPs between layers.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::graph::Graph;
use crate::layers::{GinLayer, GnnLayer};
use crate::readout::global_add_pool;

/// Multi-layer GIN Model.
pub struct GinModel {
    pub layers: Vec<GinLayer>,
}

impl GinModel {
    pub fn new(in_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();
        let mut curr_in = in_dim;
        for _ in 0..num_layers {
            layers.push(GinLayer::new(curr_in, hidden_dim));
            curr_in = hidden_dim;
        }
        Self { layers }
    }

    pub fn forward_node(&self, graph: &Graph) -> Tensor {
        let mut h = graph.node_features.clone();
        for layer in &self.layers {
            h = layer.forward(graph, &h);
        }
        h
    }

    pub fn forward_graph(&self, graph: &Graph) -> Tensor {
        let node_h = self.forward_node(graph);
        let batch = vec![0; graph.num_nodes];
        global_add_pool(&node_h, &batch, 1)
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
