//! # GNN Models
//!
//! High-level stacked GNN models: GcnModel, GatModel, SageModel.
#![allow(missing_docs)]

pub mod edge_model;
pub mod gin_model;

pub use edge_model::{EdgeClassifier, EdgeRegressor};
pub use gin_model::GinModel;

use crate::graph::Graph;
use crate::layers::{GatLayer, GcnLayer, GnnLayer, SageLayer};
use crate::readout::global_mean_pool;
use brain_core::Tensor;

/// Stacked GCN Model.
pub struct GcnModel {
    pub layers: Vec<GcnLayer>,
    pub classifier_weight: Tensor,
}

impl GcnModel {
    pub fn new(in_dim: usize, hidden_dim: usize, out_dim: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();
        let mut curr_in = in_dim;
        for _ in 0..num_layers {
            layers.push(GcnLayer::new(curr_in, hidden_dim));
            curr_in = hidden_dim;
        }
        let classifier_weight = Tensor::zeros(vec![out_dim, hidden_dim]);
        Self {
            layers,
            classifier_weight,
        }
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
        global_mean_pool(&node_h, &batch, 1)
    }
}

/// Stacked GAT Model.
pub struct GatModel {
    pub layers: Vec<GatLayer>,
}

impl GatModel {
    pub fn new(in_dim: usize, hidden_dim: usize, num_layers: usize, num_heads: usize) -> Self {
        let mut layers = Vec::new();
        let mut curr_in = in_dim;
        for _ in 0..num_layers {
            layers.push(GatLayer::new(curr_in, hidden_dim, num_heads));
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
}

/// Stacked GraphSAGE Model.
pub struct SageModel {
    pub layers: Vec<SageLayer>,
}

impl SageModel {
    pub fn new(in_dim: usize, hidden_dim: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();
        let mut curr_in = in_dim;
        for _ in 0..num_layers {
            layers.push(SageLayer::new(curr_in, hidden_dim));
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
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
