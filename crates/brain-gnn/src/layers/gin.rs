//! # Graph Isomorphism Network (GIN) Layer
//!
//! Maximally powerful GNN layer: h_v = MLP( (1 + eps) * h_v + sum_u h_u ).
#![allow(missing_docs)]

use super::GnnLayer;
use crate::graph::Graph;
use crate::impl_::transform_node_features;
use crate::ops::aggregate_sum;
use brain_core::Tensor;

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
        Self {
            in_dim,
            out_dim,
            eps: 0.0,
            mlp_w1,
            mlp_w2,
        }
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

    fn in_dim(&self) -> usize {
        self.in_dim
    }
    fn out_dim(&self) -> usize {
        self.out_dim
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
