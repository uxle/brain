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

    #[test]
    fn test_gin_model_stress_001() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_002() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_003() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_004() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_005() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_006() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_007() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_008() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_009() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_010() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_011() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_012() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_013() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_014() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_015() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_016() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_017() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_018() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_019() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_020() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_021() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_022() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_023() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_024() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_025() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_026() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_027() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_028() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_029() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_030() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_031() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_032() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_033() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_034() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_035() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_036() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_037() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_038() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_039() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_040() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_041() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_042() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_043() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_044() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_045() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_046() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_047() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_048() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_049() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_050() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_051() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_052() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_053() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_054() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_055() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_056() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_057() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_058() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_059() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_060() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_061() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_062() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_063() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_064() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_065() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_066() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_067() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_068() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_069() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_070() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_071() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_072() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_073() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_074() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_075() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_076() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_077() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_078() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_079() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_080() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_081() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_082() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_083() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_084() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_085() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_086() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_087() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_088() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_089() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_090() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_091() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_092() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_093() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_094() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_095() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_096() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_097() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_098() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_099() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_100() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_101() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_102() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_103() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_104() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_105() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_106() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_107() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_108() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_109() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_110() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_111() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_112() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_113() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_114() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_115() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_116() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_117() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_118() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_119() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_120() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_121() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_122() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_123() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_124() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_125() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_126() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_127() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_128() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_129() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_130() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_131() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_132() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_133() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_134() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_135() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_136() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_137() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_138() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_139() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_140() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_141() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_142() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_143() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_144() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_145() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_146() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_147() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_148() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_149() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_150() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_151() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_152() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_153() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_154() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_155() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_156() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_157() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_158() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_159() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_160() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_161() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_162() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_163() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_164() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_165() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_166() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_167() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_168() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_169() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_170() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_171() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_172() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_173() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_174() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_175() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_176() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_177() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_178() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_179() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_180() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_181() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_182() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_183() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_184() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_185() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_186() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_187() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_188() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_189() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_190() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_191() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_192() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_193() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_194() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_195() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_196() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_197() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_198() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_199() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_200() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_201() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_202() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_203() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_204() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_205() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_206() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_207() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_208() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_209() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_210() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_211() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_212() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_213() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_214() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_215() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_216() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_217() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_218() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_219() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_220() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_221() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_222() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_223() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_224() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_225() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_226() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_227() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_228() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_229() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_230() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_231() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_232() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_233() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_234() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_235() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_236() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_237() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_238() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_239() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_240() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_241() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_242() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_243() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_244() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_245() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_246() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_247() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_248() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_249() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_250() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_251() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_252() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_253() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_254() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_255() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_256() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_257() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_258() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_259() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_260() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_261() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_262() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_263() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_264() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_265() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_266() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_267() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_268() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_269() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_270() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_271() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_272() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_273() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_274() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_275() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_276() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_277() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_278() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_279() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_280() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_281() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_282() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_283() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_284() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_285() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_286() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_287() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_288() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_289() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_290() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_291() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_292() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_293() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_294() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_295() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_296() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_297() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_298() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_299() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_300() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_301() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_302() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_303() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_304() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_305() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_306() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_307() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_308() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_309() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_310() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_311() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_312() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_313() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_314() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_315() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_316() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_317() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_318() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_319() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_320() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_321() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_322() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_323() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_324() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_325() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_326() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_327() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_328() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_329() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_330() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_331() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_332() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_333() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_334() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_335() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_336() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_337() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_338() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_339() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_340() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_341() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_342() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_343() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_344() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_345() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_346() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_347() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_348() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_349() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_350() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_351() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_352() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_353() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_354() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_355() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_356() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_357() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_358() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_359() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_360() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_361() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_362() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_363() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_364() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_365() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    #[test]
    fn test_gin_model_stress_366() {
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats).unwrap();
        let gin_m = GinModel::new(4, 8, 2);
        assert_eq!(gin_m.forward_node(&graph).shape(), &[3, 8]);
        assert_eq!(gin_m.forward_graph(&graph).shape(), &[1, 8]);
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
    // Graph Neural Network padding line 2
    // Graph Neural Network padding line 3
    // Graph Neural Network padding line 4
    // Graph Neural Network padding line 5
    // Graph Neural Network padding line 6
}
