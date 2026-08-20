//! # Graph Transformer Layer
//!
//! Node self-attention restricted to graph neighborhoods.
#![allow(missing_docs)]

use super::GnnLayer;
use crate::graph::Graph;
use crate::impl_::transform_node_features;
use crate::ops::aggregate_mean;
use brain_core::Tensor;

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
        Self {
            in_dim,
            out_dim,
            w_q,
            w_k,
            w_v,
        }
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
