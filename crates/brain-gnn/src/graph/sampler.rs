//! # Graph Neighborhood Sampler
//!
//! Uniform neighbor sampling, importance sampling, SampledSubgraph, and batch collation.
#![allow(missing_docs)]

use super::Graph;
use crate::core::BatchGraph;
use brain_core::Tensor;

/// Represents a sampled subgraph (for GraphSAGE/large-graph training).
#[derive(Debug, Clone)]
pub struct SampledSubgraph {
    pub target_nodes: Vec<usize>,
    pub sampled_edges_src: Vec<usize>,
    pub sampled_edges_dst: Vec<usize>,
    pub sampled_features: Tensor,
}

/// Uniformly samples at most `num_samples` neighbors for each target node.
pub fn sample_neighbors(
    graph: &Graph,
    target_nodes: &[usize],
    num_samples: usize,
    seed: u64,
) -> SampledSubgraph {
    let mut rng = seed;
    let lcg = |s: &mut u64| -> usize {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 32) as usize
    };

    let mut adj_list = vec![Vec::new(); graph.num_nodes];
    for i in 0..graph.src_nodes.len() {
        let s = graph.src_nodes[i];
        let d = graph.dst_nodes[i];
        if s < graph.num_nodes && d < graph.num_nodes {
            adj_list[s].push(d);
        }
    }

    let mut src_edges = Vec::new();
    let mut dst_edges = Vec::new();

    for &node in target_nodes {
        if node < graph.num_nodes {
            let neighbors = &adj_list[node];
            if !neighbors.is_empty() {
                let k = num_samples.min(neighbors.len());
                for _ in 0..k {
                    let idx = lcg(&mut rng) % neighbors.len();
                    src_edges.push(node);
                    dst_edges.push(neighbors[idx]);
                }
            }
        }
    }

    SampledSubgraph {
        target_nodes: target_nodes.to_vec(),
        sampled_edges_src: src_edges,
        sampled_edges_dst: dst_edges,
        sampled_features: graph.node_features.clone(),
    }
}

/// Combines a slice of individual graphs into a disjoint union `BatchGraph`.
pub fn collate_graphs(graphs: &[Graph]) -> BatchGraph {
    let mut total_src = Vec::new();
    let mut total_dst = Vec::new();
    let mut all_feat_data = Vec::new();
    let mut offsets = Vec::new();
    let mut graph_ids = Vec::new();

    let mut current_offset = 0;
    let feat_dim = if !graphs.is_empty() {
        graphs[0].feature_dim()
    } else {
        0
    };

    for (g_idx, g) in graphs.iter().enumerate() {
        offsets.push(current_offset);
        for &s in &g.src_nodes {
            total_src.push(s + current_offset);
        }
        for &d in &g.dst_nodes {
            total_dst.push(d + current_offset);
        }
        all_feat_data.extend_from_slice(&g.node_features.to_vec());
        for _ in 0..g.num_nodes {
            graph_ids.push(g_idx);
        }
        current_offset += g.num_nodes;
    }

    let total_nodes = current_offset;
    let feat_tensor = Tensor::from_vec(all_feat_data, vec![total_nodes, feat_dim]);

    BatchGraph::new(total_src, total_dst, feat_tensor, offsets, graph_ids)
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
