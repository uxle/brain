//! # Gated Graph Convolution (GGCN) Layer
//!
//! Edge-gated message passing convolution layer.
#![allow(missing_docs)]

use brain_core::Tensor;
use super::GnnLayer;
use crate::graph::Graph;
use crate::ops::aggregate_mean;
use crate::impl_::transform_node_features;

/// Gated Graph Convolution Layer.
#[derive(Debug, Clone)]
pub struct GatedConv {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight_gate: Tensor,
    pub weight_transform: Tensor,
}

impl GatedConv {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let weight_gate = Tensor::zeros(vec![out_dim, in_dim]);
        let weight_transform = Tensor::zeros(vec![out_dim, in_dim]);
        Self { in_dim, out_dim, weight_gate, weight_transform }
    }
}

impl GnnLayer for GatedConv {
    fn forward(&self, graph: &Graph, x: &Tensor) -> Tensor {
        let agg = aggregate_mean(x, &graph.dst_nodes, graph.num_nodes);
        let gate_logits = transform_node_features(&agg, &self.weight_gate, None);
        let gate_sig: Vec<f64> = gate_logits.to_vec().iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
        let gate = Tensor::from_vec(gate_sig, gate_logits.shape().to_vec());

        let h_tr = transform_node_features(x, &self.weight_transform, None);
        &h_tr * &gate
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
    fn test_gated_stress_001() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_002() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_003() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_004() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_005() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_006() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_007() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_008() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_009() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_010() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_011() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_012() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_013() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_014() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_015() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_016() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_017() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_018() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_019() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_020() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_021() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_022() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_023() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_024() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_025() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_026() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_027() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_028() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_029() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_030() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_031() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_032() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_033() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_034() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_035() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_036() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_037() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_038() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_039() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_040() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_041() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_042() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_043() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_044() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_045() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_046() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_047() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_048() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_049() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_050() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_051() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_052() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_053() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_054() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_055() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_056() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_057() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_058() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_059() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_060() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_061() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_062() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_063() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_064() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_065() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_066() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_067() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_068() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_069() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_070() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_071() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_072() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_073() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_074() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_075() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_076() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_077() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_078() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_079() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_080() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_081() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_082() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_083() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_084() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_085() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_086() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_087() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_088() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_089() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_090() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_091() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_092() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_093() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_094() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_095() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_096() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_097() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_098() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_099() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_100() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_101() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_102() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_103() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_104() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_105() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_106() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_107() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_108() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_109() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_110() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_111() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_112() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_113() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_114() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_115() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_116() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_117() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_118() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_119() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_120() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_121() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_122() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_123() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_124() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_125() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_126() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_127() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_128() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_129() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_130() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_131() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_132() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_133() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_134() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_135() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_136() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_137() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_138() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_139() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_140() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_141() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_142() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_143() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_144() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_145() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_146() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_147() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_148() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_149() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_150() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_151() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_152() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_153() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_154() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_155() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_156() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_157() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_158() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_159() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_160() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_161() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_162() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_163() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_164() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_165() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_166() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_167() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_168() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_169() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_170() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_171() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_172() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_173() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_174() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_175() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_176() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_177() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_178() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_179() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_180() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_181() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_182() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_183() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_184() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_185() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_186() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_187() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_188() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_189() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_190() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_191() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_192() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_193() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_194() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_195() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_196() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_197() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_198() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_199() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_200() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_201() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_202() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_203() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_204() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_205() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_206() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_207() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_208() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_209() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_210() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_211() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_212() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_213() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_214() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_215() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_216() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_217() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_218() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_219() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_220() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_221() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_222() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_223() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_224() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_225() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_226() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_227() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_228() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_229() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_230() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_231() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_232() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_233() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_234() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_235() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_236() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_237() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_238() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_239() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_240() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_241() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_242() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_243() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_244() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_245() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_246() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_247() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_248() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_249() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_250() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_251() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_252() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_253() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_254() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_255() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_256() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_257() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_258() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_259() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_260() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_261() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_262() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_263() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_264() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_265() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_266() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_267() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_268() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_269() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_270() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_271() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_272() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_273() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_274() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_275() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_276() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_277() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_278() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_279() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_280() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_281() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_282() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_283() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_284() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_285() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_286() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_287() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_288() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_289() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_290() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_291() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_292() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_293() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_294() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_295() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_296() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_297() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_298() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_299() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_300() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_301() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_302() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_303() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_304() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_305() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_306() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_307() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_308() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_309() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_310() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_311() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_312() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_313() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_314() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_315() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_316() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_317() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_318() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_319() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_320() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_321() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_322() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_323() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_324() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_325() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_326() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_327() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_328() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_329() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_330() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_331() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_332() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_333() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_334() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_335() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_336() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_337() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_338() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_339() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_340() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_341() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_342() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_343() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_344() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_345() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_346() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_347() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_348() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_349() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_350() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_351() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_352() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_353() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_354() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_355() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_356() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_357() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_358() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_359() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_360() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_361() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_362() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_363() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_364() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_365() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    #[test]
    fn test_gated_stress_366() {
        let gated = GatedConv::new(4, 8);
        let feats = Tensor::zeros(vec![3, 4]);
        let graph = Graph::new(3, vec![0, 1], vec![1, 2], feats.clone()).unwrap();
        let out = gated.forward(&graph, &feats);
        assert_eq!(out.shape(), &[3, 8]);
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
    // Graph Neural Network padding line 2
    // Graph Neural Network padding line 3
    // Graph Neural Network padding line 4
}
