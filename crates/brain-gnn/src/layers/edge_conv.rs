//! # EdgeConv Layer (Dynamic Graph)
//!
//! Dynamic EdgeConv layer for point clouds / KNN graphs: h_i' = max_j MLP(h_i || h_j - h_i).
#![allow(missing_docs)]

use super::GnnLayer;
use crate::graph::Graph;
use crate::impl_::transform_node_features;
use crate::ops::aggregate_max;
use brain_core::Tensor;

/// EdgeConv Layer struct.
#[derive(Debug, Clone)]
pub struct EdgeConv {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weight: Tensor,
}

impl EdgeConv {
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        let weight = Tensor::zeros(vec![out_dim, in_dim * 2]);
        Self {
            in_dim,
            out_dim,
            weight,
        }
    }
}

impl GnnLayer for EdgeConv {
    fn forward(&self, graph: &Graph, x: &Tensor) -> Tensor {
        let num_nodes = graph.num_nodes;
        let agg_max = aggregate_max(x, &graph.dst_nodes, num_nodes);
        let diff = &agg_max - x;

        // Concatenate x and diff
        let x_v = x.to_vec();
        let d_v = diff.to_vec();
        let combined: Vec<f64> = x_v.into_iter().chain(d_v).collect();
        let _combined_len = combined.len();

        let h_cat = Tensor::from_vec(combined, vec![num_nodes, self.in_dim * 2]);
        transform_node_features(&h_cat, &self.weight, None)
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
