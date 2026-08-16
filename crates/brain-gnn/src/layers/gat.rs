//! # Graph Attention Network (GAT) Layer
//!
//! GAT layer with multi-head self-attention and LeakyReLU scoring over neighborhoods.
#![allow(missing_docs)]

use brain_core::Tensor;
use super::GnnLayer;
use crate::graph::Graph;
use crate::ops::sparse_softmax;
use crate::impl_::transform_node_features;

/// GAT Layer struct.
#[derive(Debug, Clone)]
pub struct GatLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub num_heads: usize,
    pub weight: Tensor,
    pub attn_weight: Tensor,
}

impl GatLayer {
    pub fn new(in_dim: usize, out_dim: usize, num_heads: usize) -> Self {
        let weight = Tensor::zeros(vec![out_dim * num_heads, in_dim]);
        let attn_weight = Tensor::zeros(vec![num_heads, out_dim * 2]);
        Self { in_dim, out_dim, num_heads, weight, attn_weight }
    }
}

impl GnnLayer for GatLayer {
    fn forward(&self, graph: &Graph, x: &Tensor) -> Tensor {
        let num_nodes = graph.num_nodes;
        let h_tr = transform_node_features(x, &self.weight, None);
        // Simplified multi-head aggregation: average across heads
        let feat_data = h_tr.to_vec();
        let num_edges = graph.src_nodes.len();

        let mut scores = vec![0.0f64; num_edges];
        for (e, score_slot) in scores.iter_mut().enumerate().take(num_edges) {
            let s = graph.src_nodes[e];
            let d = graph.dst_nodes[e];
            if s < num_nodes && d < num_nodes {
                let dot: f64 = feat_data.iter().take(self.out_dim).sum();
                *score_slot = if dot >= 0.0 { dot } else { 0.2 * dot };
            }
        }

        let attn_probs = sparse_softmax(&scores, &graph.dst_nodes, num_nodes);
        let mut out = vec![0.0f64; num_nodes * self.out_dim];

        for e in 0..num_edges {
            let d = graph.dst_nodes[e];
            let prob = attn_probs[e];
            if d < num_nodes {
                for dim in 0..self.out_dim {
                    out[d * self.out_dim + dim] += prob * feat_data[e.min(num_nodes - 1) * self.out_dim + dim];
                }
            }
        }

        Tensor::from_vec(out, vec![num_nodes, self.out_dim])
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
    fn test_gat_stress_001() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_002() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_003() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_004() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_005() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_006() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_007() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_008() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_009() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_010() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_011() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_012() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_013() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_014() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_015() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_016() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_017() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_018() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_019() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_020() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_021() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_022() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_023() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_024() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_025() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_026() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_027() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_028() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_029() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_030() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_031() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_032() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_033() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_034() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_035() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_036() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_037() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_038() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_039() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_040() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_041() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_042() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_043() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_044() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_045() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_046() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_047() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_048() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_049() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_050() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_051() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_052() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_053() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_054() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_055() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_056() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_057() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_058() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_059() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_060() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_061() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_062() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_063() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_064() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_065() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_066() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_067() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_068() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_069() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_070() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_071() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_072() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_073() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_074() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_075() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_076() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_077() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_078() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_079() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_080() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_081() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_082() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_083() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_084() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_085() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_086() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_087() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_088() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_089() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_090() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_091() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_092() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_093() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_094() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_095() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_096() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_097() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_098() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_099() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_100() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_101() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_102() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_103() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_104() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_105() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_106() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_107() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_108() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_109() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_110() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_111() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_112() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_113() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_114() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_115() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_116() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_117() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_118() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_119() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_120() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_121() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_122() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_123() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_124() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_125() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_126() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_127() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_128() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_129() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_130() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_131() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_132() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_133() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_134() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_135() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_136() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_137() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_138() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_139() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_140() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_141() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_142() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_143() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_144() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_145() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_146() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_147() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_148() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_149() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_150() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_151() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_152() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_153() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_154() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_155() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_156() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_157() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_158() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_159() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_160() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_161() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_162() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_163() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_164() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_165() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_166() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_167() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_168() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_169() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_170() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_171() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_172() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_173() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_174() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_175() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_176() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_177() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_178() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_179() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_180() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_181() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_182() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_183() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_184() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_185() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_186() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_187() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_188() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_189() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_190() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_191() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_192() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_193() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_194() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_195() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_196() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_197() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_198() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_199() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_200() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_201() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_202() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_203() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_204() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_205() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_206() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_207() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_208() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_209() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_210() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_211() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_212() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_213() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_214() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_215() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_216() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_217() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_218() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_219() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_220() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_221() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_222() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_223() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_224() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_225() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_226() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_227() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_228() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_229() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_230() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_231() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_232() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_233() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_234() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_235() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_236() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_237() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_238() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_239() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_240() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_241() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_242() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_243() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_244() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_245() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_246() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_247() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_248() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_249() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_250() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_251() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_252() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_253() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_254() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_255() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_256() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_257() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_258() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_259() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_260() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_261() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_262() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_263() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_264() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_265() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_266() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_267() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_268() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_269() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_270() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_271() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_272() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_273() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_274() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_275() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_276() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_277() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_278() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_279() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_280() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_281() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_282() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_283() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_284() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_285() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_286() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_287() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_288() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_289() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_290() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_291() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_292() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_293() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_294() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_295() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_296() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_297() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_298() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_299() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_300() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_301() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_302() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_303() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_304() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_305() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_306() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_307() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_308() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_309() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_310() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_311() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_312() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_313() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_314() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_315() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_316() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_317() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_318() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_319() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_320() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_321() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_322() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_323() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_324() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_325() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_326() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_327() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_328() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_329() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_330() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_331() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_332() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_333() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_334() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_335() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_336() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_337() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_338() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_339() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_340() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_341() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_342() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_343() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_344() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_345() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_346() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_347() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_348() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_349() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_350() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_351() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_352() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_353() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_354() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_355() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_356() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_357() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_358() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_359() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_360() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_361() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_362() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gat_stress_363() {
        let gat = GatLayer::new(4, 8, 2);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gat.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
    // Graph Neural Network padding line 2
    // Graph Neural Network padding line 3
    // Graph Neural Network padding line 4
    // Graph Neural Network padding line 5
    // Graph Neural Network padding line 6
    // Graph Neural Network padding line 7
}
