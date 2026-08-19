//! # GNN Implementation Utilities
//!
//! Execution wrappers: `forward_node`, `forward_graph`, `embed`, `predict`.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::graph::Graph;

/// Simple linear transformation of node features: X * W + b.
pub fn transform_node_features(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let x_data = x.to_vec();
    let w_data = weight.to_vec();
    let num_nodes = x.shape()[0];
    let in_dim = if x.shape().len() > 1 { x.shape()[1] } else { 1 };
    let out_dim = weight.shape()[0];

    let mut out = vec![0.0f64; num_nodes * out_dim];
    for n in 0..num_nodes {
        for o in 0..out_dim {
            let mut s = 0.0f64;
            for i in 0..in_dim {
                s += x_data[n * in_dim + i] * w_data[o * in_dim + i];
            }
            if let Some(b) = bias {
                s += b.to_vec().get(o).copied().unwrap_or(0.0);
            }
            out[n * out_dim + o] = s;
        }
    }
    Tensor::from_vec(out, vec![num_nodes, out_dim])
}

/// Helper function to compute node embeddings for a graph.
pub fn embed_nodes(graph: &Graph, weights: &[Tensor]) -> Tensor {
    let mut h = graph.node_features.clone();
    for w in weights {
        h = transform_node_features(&h, w, None);
    }
    h
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
