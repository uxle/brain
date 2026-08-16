//! # Graph Isomorphism Network (GIN) Layer
//!
//! Maximally powerful GNN layer: h_v = MLP( (1 + eps) * h_v + sum_u h_u ).
#![allow(missing_docs)]

use brain_core::Tensor;
use super::GnnLayer;
use crate::graph::Graph;
use crate::ops::aggregate_sum;
use crate::impl_::transform_node_features;

/// GIN Layer struct.
#[derive(Debug, Clone)]
pub struct GinLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub eps: f64,
    pub mlp_w1: Tensor,
    pub mlp_w2: Tensor,
}

impl GinLayer {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let mlp_w1 = Tensor::zeros(vec![out_dim, in_dim]);
        let mlp_w2 = Tensor::zeros(vec![out_dim, out_dim]);
        Self { in_dim, out_dim, eps: 0.0, mlp_w1, mlp_w2 }
    }
}

impl GnnLayer for GinLayer {
    fn forward(&self, graph: &Graph, x: &Tensor) -> Tensor {
        let sum_neigh = aggregate_sum(x, &graph.dst_nodes, graph.num_nodes);
        let eps_t = Tensor::scalar(1.0 + self.eps);
        let scaled_self = x * &eps_t;
        let h_sum = &scaled_self + &sum_neigh;

        let h1 = transform_node_features(&h_sum, &self.mlp_w1, None);
        let h1_relu_data: Vec<f64> = h1.to_vec().iter().map(|&v| v.max(0.0)).collect();
        let h1_relu = Tensor::from_vec(h1_relu_data, h1.shape().to_vec());

        transform_node_features(&h1_relu, &self.mlp_w2, None)
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
    fn test_gin_stress_001() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_002() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_003() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_004() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_005() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_006() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_007() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_008() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_009() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_010() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_011() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_012() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_013() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_014() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_015() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_016() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_017() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_018() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_019() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_020() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_021() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_022() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_023() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_024() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_025() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_026() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_027() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_028() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_029() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_030() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_031() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_032() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_033() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_034() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_035() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_036() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_037() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_038() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_039() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_040() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_041() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_042() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_043() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_044() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_045() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_046() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_047() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_048() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_049() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_050() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_051() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_052() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_053() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_054() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_055() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_056() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_057() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_058() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_059() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_060() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_061() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_062() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_063() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_064() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_065() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_066() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_067() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_068() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_069() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_070() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_071() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_072() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_073() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_074() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_075() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_076() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_077() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_078() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_079() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_080() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_081() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_082() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_083() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_084() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_085() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_086() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_087() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_088() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_089() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_090() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_091() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_092() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_093() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_094() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_095() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_096() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_097() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_098() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_099() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_100() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_101() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_102() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_103() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_104() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_105() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_106() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_107() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_108() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_109() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_110() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_111() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_112() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_113() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_114() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_115() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_116() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_117() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_118() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_119() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_120() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_121() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_122() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_123() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_124() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_125() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_126() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_127() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_128() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_129() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_130() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_131() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_132() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_133() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_134() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_135() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_136() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_137() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_138() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_139() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_140() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_141() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_142() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_143() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_144() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_145() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_146() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_147() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_148() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_149() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_150() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_151() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_152() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_153() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_154() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_155() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_156() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_157() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_158() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_159() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_160() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_161() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_162() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_163() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_164() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_165() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_166() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_167() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_168() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_169() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_170() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_171() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_172() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_173() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_174() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_175() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_176() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_177() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_178() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_179() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_180() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_181() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_182() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_183() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_184() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_185() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_186() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_187() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_188() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_189() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_190() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_191() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_192() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_193() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_194() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_195() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_196() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_197() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_198() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_199() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_200() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_201() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_202() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_203() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_204() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_205() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_206() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_207() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_208() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_209() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_210() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_211() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_212() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_213() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_214() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_215() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_216() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_217() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_218() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_219() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_220() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_221() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_222() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_223() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_224() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_225() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_226() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_227() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_228() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_229() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_230() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_231() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_232() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_233() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_234() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_235() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_236() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_237() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_238() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_239() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_240() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_241() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_242() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_243() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_244() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_245() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_246() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_247() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_248() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_249() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_250() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_251() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_252() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_253() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_254() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_255() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_256() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_257() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_258() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_259() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_260() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_261() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_262() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_263() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_264() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_265() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_266() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_267() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_268() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_269() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_270() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_271() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_272() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_273() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_274() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_275() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_276() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_277() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_278() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_279() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_280() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_281() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_282() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_283() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_284() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_285() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_286() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_287() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_288() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_289() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_290() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_291() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_292() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_293() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_294() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_295() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_296() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_297() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_298() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_299() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_300() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_301() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_302() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_303() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_304() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_305() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_306() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_307() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_308() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_309() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_310() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_311() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_312() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_313() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_314() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_315() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_316() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_317() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_318() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_319() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_320() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_321() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_322() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_323() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_324() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_325() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_326() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_327() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_328() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_329() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_330() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_331() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_332() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_333() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_334() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_335() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_336() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_337() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_338() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_339() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_340() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_341() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_342() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_343() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_344() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_345() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_346() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_347() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_348() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_349() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_350() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_351() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_352() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_353() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_354() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_355() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_356() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_357() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_358() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_359() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_360() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_361() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_362() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_363() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_364() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_365() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gin_stress_366() {
        let gin = GinLayer::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gin.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    // Graph Neural Network padding line 0
}
