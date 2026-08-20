//! # Graph Operations
//!
//! Subgraph extraction, degree computation, self-loops, and dense conversion.
#![allow(missing_docs)]

use super::Graph;
use crate::ops::normalize_adj;
use brain_core::Tensor;

/// Computes the in-degrees of all nodes in a graph.
pub fn in_degrees(graph: &Graph) -> Vec<usize> {
    let mut deg = vec![0usize; graph.num_nodes];
    for &d in &graph.dst_nodes {
        if d < graph.num_nodes {
            deg[d] += 1;
        }
    }
    deg
}

/// Computes the out-degrees of all nodes in a graph.
pub fn out_degrees(graph: &Graph) -> Vec<usize> {
    graph.degrees()
}

/// Converts a graph edge list to a dense adjacency matrix tensor [N, N].
pub fn to_dense_adj(graph: &Graph) -> Tensor {
    let n = graph.num_nodes;
    let mut adj = vec![0.0f64; n * n];
    for i in 0..graph.src_nodes.len() {
        let s = graph.src_nodes[i];
        let d = graph.dst_nodes[i];
        let w = graph
            .edge_weights
            .as_ref()
            .map(|weights| weights[i])
            .unwrap_or(1.0);
        if s < n && d < n {
            adj[s * n + d] += w;
        }
    }
    Tensor::from_vec(adj, vec![n, n])
}

/// Computes normalized adjacency for a `Graph`.
pub fn normalized_graph_adj(graph: &Graph) -> Tensor {
    let dense = to_dense_adj(graph);
    normalize_adj(&dense)
}

/// Extracts an induced subgraph given node indices.
pub fn induced_subgraph(graph: &Graph, node_subset: &[usize]) -> Graph {
    let mut is_in_subset = vec![false; graph.num_nodes];
    let mut new_id = vec![0usize; graph.num_nodes];
    for (idx, &n) in node_subset.iter().enumerate() {
        if n < graph.num_nodes {
            is_in_subset[n] = true;
            new_id[n] = idx;
        }
    }

    let sub_num_nodes = node_subset.len();
    let mut sub_src = Vec::new();
    let mut sub_dst = Vec::new();

    for i in 0..graph.src_nodes.len() {
        let s = graph.src_nodes[i];
        let d = graph.dst_nodes[i];
        if s < graph.num_nodes && d < graph.num_nodes && is_in_subset[s] && is_in_subset[d] {
            sub_src.push(new_id[s]);
            sub_dst.push(new_id[d]);
        }
    }

    let feat_dim = graph.feature_dim();
    let old_feats = graph.node_features.to_vec();
    let mut sub_feats = vec![0.0f64; sub_num_nodes * feat_dim];

    for (new_idx, &old_idx) in node_subset.iter().enumerate() {
        if old_idx < graph.num_nodes {
            for dim in 0..feat_dim {
                sub_feats[new_idx * feat_dim + dim] = old_feats[old_idx * feat_dim + dim];
            }
        }
    }

    Graph::new(
        sub_num_nodes,
        sub_src,
        sub_dst,
        Tensor::from_vec(sub_feats, vec![sub_num_nodes, feat_dim]),
    )
    .unwrap()
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
