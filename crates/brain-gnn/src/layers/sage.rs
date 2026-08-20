//! # GraphSAGE Layer
//!
//! Neighborhood aggregation with self-connection concatenation: h_v = W * [h_v || agg(h_u)].
#![allow(missing_docs)]

use super::GnnLayer;
use crate::graph::Graph;
use crate::impl_::transform_node_features;
use crate::ops::aggregate_mean;
use brain_core::Tensor;

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
        Self {
            in_dim,
            out_dim,
            weight_self,
            weight_neigh,
        }
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
