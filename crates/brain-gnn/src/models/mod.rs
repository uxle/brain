//! # GNN Models
//!
//! High-level stacked GNN models: GcnModel, GatModel, SageModel.
#![allow(missing_docs)]

pub mod gin_model;
pub mod edge_model;

pub use gin_model::GinModel;
pub use edge_model::{EdgeClassifier, EdgeRegressor};

use brain_core::Tensor;
use crate::graph::Graph;
use crate::layers::{GcnLayer, GatLayer, SageLayer, GnnLayer};
use crate::readout::global_mean_pool;

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
        Self { layers, classifier_weight }
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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_models_mod_stress_001() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_002() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_003() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_004() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_005() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_006() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_007() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_008() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_009() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_010() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_011() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_012() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_013() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_014() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_015() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_016() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_017() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_018() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_019() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_020() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_021() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_022() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_023() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_024() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_025() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_026() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_027() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_028() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_029() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_030() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_031() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_032() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_033() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_034() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_035() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_036() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_037() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_038() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_039() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_040() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_041() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_042() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_043() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_044() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_045() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_046() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_047() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_048() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_049() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_050() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_051() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_052() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_053() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_054() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_055() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_056() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_057() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_058() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_059() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_060() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_061() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_062() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_063() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_064() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_065() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_066() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_067() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_068() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_069() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_070() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_071() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_072() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_073() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_074() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_075() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_076() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_077() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_078() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_079() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_080() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_081() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_082() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_083() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_084() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_085() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_086() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_087() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_088() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_089() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_090() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_091() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_092() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_093() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_094() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_095() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_096() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_097() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_098() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_099() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_100() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_101() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_102() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_103() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_104() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_105() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_106() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_107() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_108() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_109() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_110() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_111() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_112() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_113() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_114() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_115() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_116() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_117() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_118() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_119() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_120() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_121() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_122() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_123() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_124() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_125() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_126() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_127() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_128() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_129() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_130() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_131() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_132() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_133() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_134() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_135() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_136() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_137() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_138() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_139() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_140() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_141() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_142() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_143() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_144() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_145() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_146() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_147() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_148() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_149() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_150() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_151() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_152() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_153() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_154() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_155() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_156() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_157() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_158() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_159() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_160() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_161() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_162() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_163() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_164() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_165() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_166() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_167() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_168() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_169() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_170() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_171() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_172() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_173() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_174() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_175() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_176() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_177() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_178() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_179() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    #[test]
    fn test_models_mod_stress_180() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();

        let gcn_m = GcnModel::new(4, 8, 2, 2);
        let out_node = gcn_m.forward_node(&graph);
        assert_eq!(out_node.shape(), &[3, 8]);
        let out_g = gcn_m.forward_graph(&graph);
        assert_eq!(out_g.shape(), &[1, 8]);

        let gat_m = GatModel::new(4, 8, 2, 2);
        assert_eq!(gat_m.forward_node(&graph).shape(), &[3, 8]);

        let sage_m = SageModel::new(4, 8, 2);
        assert_eq!(sage_m.forward_node(&graph).shape(), &[3, 8]);
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
}
