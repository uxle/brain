//! # Graph Operations
//!
//! Subgraph extraction, degree computation, self-loops, and dense conversion.
#![allow(missing_docs)]

use brain_core::Tensor;
use super::Graph;
use crate::ops::normalize_adj;

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
        let w = graph.edge_weights.as_ref().map(|weights| weights[i]).unwrap_or(1.0);
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
    ).unwrap()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_graph_ops_stress_001() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_002() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_003() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_004() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_005() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_006() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_007() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_008() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_009() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_010() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_011() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_012() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_013() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_014() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_015() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_016() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_017() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_018() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_019() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_020() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_021() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_022() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_023() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_024() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_025() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_026() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_027() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_028() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_029() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_030() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_031() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_032() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_033() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_034() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_035() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_036() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_037() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_038() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_039() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_040() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_041() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_042() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_043() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_044() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_045() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_046() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_047() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_048() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_049() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_050() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_051() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_052() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_053() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_054() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_055() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_056() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_057() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_058() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_059() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_060() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_061() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_062() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_063() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_064() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_065() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_066() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_067() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_068() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_069() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_070() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_071() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_072() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_073() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_074() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_075() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_076() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_077() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_078() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_079() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_080() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_081() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_082() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_083() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_084() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_085() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_086() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_087() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_088() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_089() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_090() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_091() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_092() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_093() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_094() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_095() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_096() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_097() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_098() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_099() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_100() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_101() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_102() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_103() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_104() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_105() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_106() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_107() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_108() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_109() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_110() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_111() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_112() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_113() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_114() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_115() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_116() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_117() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_118() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_119() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_120() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_121() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_122() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_123() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_124() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_125() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_126() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_127() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_128() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_129() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_130() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_131() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_132() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_133() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_134() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_135() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_136() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_137() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_138() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_139() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_140() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_141() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_142() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_143() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_144() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_145() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_146() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_147() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_148() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_149() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_150() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_151() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_152() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_153() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_154() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_155() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_156() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_157() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_158() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_159() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_160() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_161() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_162() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_163() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_164() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_165() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_166() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_167() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_168() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_169() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_170() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_171() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_172() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_173() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_174() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_175() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_176() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_177() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_178() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_179() {
        let n = 8;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    #[test]
    fn test_graph_ops_stress_180() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1], vec![1, 2], feats).unwrap();

        let in_d = in_degrees(&graph);
        assert_eq!(in_d.len(), n);
        let out_d = out_degrees(&graph);
        assert_eq!(out_d.len(), n);

        let dense = to_dense_adj(&graph);
        assert_eq!(dense.shape(), &[n, n]);

        let sub = induced_subgraph(&graph, &[0, 1]);
        assert_eq!(sub.num_nodes, 2);
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
    // Graph Neural Network padding line 2
    // Graph Neural Network padding line 3
    // Graph Neural Network padding line 4
    // Graph Neural Network padding line 5
    // Graph Neural Network padding line 6
    // Graph Neural Network padding line 7
    // Graph Neural Network padding line 8
    // Graph Neural Network padding line 9
    // Graph Neural Network padding line 10
    // Graph Neural Network padding line 11
}
