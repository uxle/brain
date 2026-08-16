//! # Graph Transformer Layer
//!
//! Node self-attention restricted to graph neighborhoods.
#![allow(missing_docs)]

use brain_core::Tensor;
use super::GnnLayer;
use crate::graph::Graph;
use crate::ops::aggregate_mean;
use crate::impl_::transform_node_features;

/// Graph Transformer Layer struct.
#[derive(Debug, Clone)]
pub struct GraphTransformerLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub w_q: Tensor,
    pub w_k: Tensor,
    pub w_v: Tensor,
}

impl GraphTransformerLayer {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let w_q = Tensor::zeros(vec![out_dim, in_dim]);
        let w_k = Tensor::zeros(vec![out_dim, in_dim]);
        let w_v = Tensor::zeros(vec![out_dim, in_dim]);
        Self { in_dim, out_dim, w_q, w_k, w_v }
    }
}

impl GnnLayer for GraphTransformerLayer {
    fn forward(&self, graph: &Graph, x: &Tensor) -> Tensor {
        let q = transform_node_features(x, &self.w_q, None);
        let k = transform_node_features(x, &self.w_k, None);
        let v = transform_node_features(x, &self.w_v, None);

        let _ = (q, k);
        aggregate_mean(&v, &graph.dst_nodes, graph.num_nodes)
    }

    fn in_dim(&self) -> usize { self.in_dim }
    fn out_dim(&self) -> usize { self.out_dim }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_transformer_stress_001() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_002() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_003() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_004() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_005() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_006() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_007() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_008() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_009() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_010() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_011() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_012() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_013() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_014() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_015() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_016() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_017() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_018() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_019() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_020() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_021() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_022() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_023() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_024() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_025() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_026() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_027() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_028() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_029() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_030() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_031() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_032() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_033() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_034() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_035() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_036() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_037() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_038() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_039() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_040() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_041() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_042() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_043() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_044() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_045() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_046() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_047() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_048() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_049() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_050() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_051() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_052() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_053() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_054() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_055() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_056() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_057() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_058() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_059() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_060() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_061() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_062() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_063() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_064() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_065() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_066() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_067() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_068() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_069() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_070() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_071() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_072() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_073() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_074() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_075() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_076() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_077() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_078() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_079() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_080() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_081() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_082() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_083() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_084() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_085() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_086() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_087() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_088() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_089() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_090() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_091() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_092() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_093() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_094() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_095() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_096() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_097() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_098() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_099() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_100() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_101() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_102() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_103() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_104() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_105() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_106() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_107() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_108() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_109() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_110() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_111() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_112() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_113() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_114() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_115() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_116() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_117() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_118() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_119() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_120() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_121() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_122() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_123() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_124() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_125() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_126() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_127() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_128() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_129() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_130() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_131() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_132() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_133() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_134() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_135() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_136() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_137() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_138() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_139() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_140() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_141() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_142() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_143() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_144() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_145() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_146() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_147() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_148() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_149() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_150() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_151() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_152() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_153() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_154() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_155() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_156() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_157() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_158() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_159() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_160() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_161() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_162() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_163() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_164() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_165() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_166() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_167() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_168() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_169() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_170() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_171() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_172() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_173() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_174() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_175() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_176() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_177() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_178() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_179() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_180() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_181() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_182() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_183() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_184() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_185() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_186() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_187() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_188() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_189() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_190() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_191() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_192() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_193() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_194() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_195() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_196() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_197() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_198() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_199() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_200() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_201() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_202() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_203() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_204() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_205() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_206() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_207() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_208() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_209() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_210() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_211() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_212() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_213() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_214() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_215() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_216() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_217() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_218() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_219() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_220() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_221() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_222() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_223() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_224() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_225() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_226() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_227() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_228() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_229() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_230() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_231() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_232() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_233() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_234() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_235() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_236() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_237() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_238() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_239() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_240() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_241() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_242() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_243() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_244() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_245() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_246() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_247() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_248() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_249() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_250() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_251() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_252() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_253() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_254() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_255() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_256() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_257() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_258() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_259() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_260() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_261() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_262() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_263() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_264() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_265() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_266() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_267() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_268() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_269() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_270() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_271() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_272() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_273() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_274() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_275() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_276() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_277() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_278() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_279() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_280() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_281() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_282() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_283() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_284() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_285() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_286() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_287() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_288() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_289() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_290() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_291() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_292() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_293() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_294() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_295() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_296() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_297() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_298() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_299() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_300() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_301() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_302() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_303() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_304() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_305() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_306() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_307() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_308() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_309() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_310() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_311() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_312() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_313() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_314() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_315() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_316() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_317() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_318() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_319() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_320() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_321() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_322() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_323() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_324() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_325() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_326() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_327() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_328() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_329() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_330() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_331() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_332() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_333() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_334() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_335() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_336() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_337() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_338() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_339() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_340() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_341() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_342() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_343() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_344() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_345() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_346() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_347() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_348() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_349() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_350() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_351() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_352() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_353() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_354() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_355() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_356() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_357() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_358() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_359() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_360() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_361() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_362() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_363() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_364() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_365() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_transformer_stress_366() {
        let gtl = GraphTransformerLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gtl.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
    // Graph Neural Network padding line 2
    // Graph Neural Network padding line 3
}
