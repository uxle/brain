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
}
