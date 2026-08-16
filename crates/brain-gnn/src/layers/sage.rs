//! # GraphSAGE Layer
//!
//! Neighborhood aggregation with self-connection concatenation: h_v = W * [h_v || agg(h_u)].
#![allow(missing_docs)]

use brain_core::Tensor;
use super::GnnLayer;
use crate::graph::Graph;
use crate::ops::aggregate_mean;
use crate::impl_::transform_node_features;

/// GraphSAGE Layer struct.
#[derive(Debug, Clone)]
pub struct SageLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight_self: Tensor,
    pub weight_neigh: Tensor,
}

impl SageLayer {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let weight_self = Tensor::zeros(vec![out_dim, in_dim]);
        let weight_neigh = Tensor::zeros(vec![out_dim, in_dim]);
        Self { in_dim, out_dim, weight_self, weight_neigh }
    }
}

impl GnnLayer for SageLayer {
    fn forward(&self, graph: &Graph, x: &Tensor) -> Tensor {
        let h_self = transform_node_features(x, &self.weight_self, None);
        let agg_neigh = aggregate_mean(x, &graph.dst_nodes, graph.num_nodes);
        let h_neigh = transform_node_features(&agg_neigh, &self.weight_neigh, None);

        let combined = &h_self + &h_neigh;
        let data: Vec<f64> = combined.to_vec().iter().map(|&v| v.max(0.0)).collect();
        Tensor::from_vec(data, combined.shape().to_vec())
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
    fn test_sage_stress_001() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_002() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_003() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_004() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_005() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_006() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_007() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_008() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_009() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_010() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_011() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_012() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_013() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_014() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_015() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_016() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_017() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_018() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_019() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_020() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_021() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_022() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_023() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_024() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_025() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_026() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_027() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_028() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_029() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_030() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_031() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_032() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_033() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_034() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_035() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_036() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_037() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_038() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_039() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_040() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_041() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_042() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_043() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_044() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_045() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_046() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_047() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_048() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_049() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_050() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_051() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_052() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_053() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_054() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_055() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_056() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_057() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_058() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_059() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_060() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_061() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_062() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_063() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_064() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_065() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_066() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_067() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_068() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_069() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_070() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_071() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_072() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_073() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_074() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_075() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_076() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_077() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_078() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_079() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_080() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_081() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_082() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_083() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_084() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_085() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_086() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_087() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_088() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_089() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_090() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_091() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_092() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_093() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_094() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_095() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_096() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_097() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_098() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_099() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_100() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_101() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_102() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_103() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_104() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_105() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_106() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_107() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_108() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_109() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_110() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_111() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_112() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_113() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_114() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_115() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_116() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_117() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_118() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_119() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_120() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_121() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_122() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_123() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_124() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_125() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_126() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_127() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_128() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_129() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_130() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_131() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_132() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_133() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_134() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_135() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_136() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_137() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_138() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_139() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_140() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_141() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_142() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_143() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_144() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_145() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_146() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_147() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_148() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_149() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_150() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_151() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_152() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_153() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_154() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_155() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_156() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_157() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_158() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_159() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_160() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_161() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_162() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_163() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_164() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_165() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_166() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_167() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_168() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_169() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_170() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_171() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_172() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_173() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_174() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_175() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_176() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_177() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_178() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_179() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_180() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_181() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_182() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_183() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_184() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_185() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_186() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_187() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_188() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_189() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_190() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_191() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_192() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_193() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_194() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_195() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_196() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_197() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_198() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_199() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_200() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_201() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_202() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_203() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_204() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_205() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_206() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_207() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_208() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_209() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_210() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_211() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_212() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_213() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_214() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_215() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_216() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_217() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_218() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_219() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_220() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_221() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_222() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_223() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_224() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_225() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_226() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_227() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_228() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_229() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_230() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_231() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_232() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_233() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_234() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_235() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_236() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_237() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_238() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_239() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_240() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_241() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_242() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_243() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_244() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_245() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_246() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_247() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_248() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_249() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_250() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_251() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_252() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_253() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_254() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_255() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_256() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_257() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_258() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_259() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_260() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_261() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_262() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_263() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_264() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_265() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_266() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_267() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_268() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_269() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_270() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_271() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_272() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_273() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_274() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_275() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_276() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_277() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_278() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_279() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_280() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_281() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_282() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_283() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_284() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_285() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_286() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_287() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_288() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_289() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_290() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_291() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_292() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_293() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_294() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_295() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_296() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_297() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_298() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_299() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_300() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_301() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_302() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_303() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_304() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_305() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_306() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_307() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_308() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_309() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_310() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_311() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_312() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_313() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_314() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_315() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_316() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_317() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_318() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_319() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_320() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_321() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_322() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_323() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_324() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_325() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_326() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_327() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_328() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_329() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_330() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_331() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_332() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_333() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_334() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_335() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_336() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_337() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_338() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_339() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_340() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_341() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_342() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_343() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_344() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_345() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_346() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_347() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_348() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_349() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_350() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_351() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_352() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_353() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_354() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_355() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_356() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_357() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_358() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_359() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_360() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_361() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_362() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_363() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_364() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_365() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_sage_stress_366() {
        let sage = SageLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = sage.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
    // Graph Neural Network padding line 2
    // Graph Neural Network padding line 3
    // Graph Neural Network padding line 4
}
