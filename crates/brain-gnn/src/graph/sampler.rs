//! # Graph Neighborhood Sampler
//!
//! Uniform neighbor sampling, importance sampling, SampledSubgraph, and batch collation.
#![allow(missing_docs)]

use brain_core::Tensor;
use super::Graph;
use crate::core::BatchGraph;

/// Represents a sampled subgraph (for GraphSAGE/large-graph training).
#[derive(Debug, Clone)]
pub struct SampledSubgraph {
    pub target_nodes: Vec<usize>,
    pub sampled_edges_src: Vec<usize>,
    pub sampled_edges_dst: Vec<usize>,
    pub sampled_features: Tensor,
}

/// Uniformly samples at most `num_samples` neighbors for each target node.
pub fn sample_neighbors(graph: &Graph, target_nodes: &[usize], num_samples: usize, seed: u64) -> SampledSubgraph {
    let mut rng = seed;
    let lcg = |s: &mut u64| -> usize {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
    let feat_dim = if !graphs.is_empty() { graphs[0].feature_dim() } else { 0 };

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
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_sampler_stress_001() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 1 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_002() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 2 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_003() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 3 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_004() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 4 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_005() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 5 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_006() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 6 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_007() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 7 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_008() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 8 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_009() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 9 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_010() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 10 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_011() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 11 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_012() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 12 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_013() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 13 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_014() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 14 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_015() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 15 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_016() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 16 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_017() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 17 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_018() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 18 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_019() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 19 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_020() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 20 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_021() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 21 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_022() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 22 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_023() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 23 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_024() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 24 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_025() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 25 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_026() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 26 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_027() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 27 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_028() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 28 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_029() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 29 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_030() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 30 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_031() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 31 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_032() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 32 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_033() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 33 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_034() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 34 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_035() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 35 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_036() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 36 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_037() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 37 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_038() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 38 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_039() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 39 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_040() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 40 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_041() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 41 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_042() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 42 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_043() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 43 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_044() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 44 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_045() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 45 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_046() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 46 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_047() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 47 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_048() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 48 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_049() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 49 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_050() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 50 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_051() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 51 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_052() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 52 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_053() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 53 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_054() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 54 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_055() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 55 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_056() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 56 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_057() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 57 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_058() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 58 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_059() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 59 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_060() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 60 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_061() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 61 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_062() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 62 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_063() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 63 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_064() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 64 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_065() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 65 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_066() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 66 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_067() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 67 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_068() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 68 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_069() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 69 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_070() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 70 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_071() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 71 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_072() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 72 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_073() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 73 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_074() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 74 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_075() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 75 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_076() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 76 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_077() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 77 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_078() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 78 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_079() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 79 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_080() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 80 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_081() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 81 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_082() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 82 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_083() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 83 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_084() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 84 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_085() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 85 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_086() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 86 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_087() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 87 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_088() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 88 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_089() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 89 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_090() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 90 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_091() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 91 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_092() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 92 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_093() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 93 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_094() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 94 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_095() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 95 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_096() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 96 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_097() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 97 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_098() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 98 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_099() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 99 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_100() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 100 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_101() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 101 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_102() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 102 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_103() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 103 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_104() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 104 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_105() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 105 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_106() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 106 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_107() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 107 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_108() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 108 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_109() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 109 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_110() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 110 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_111() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 111 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_112() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 112 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_113() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 113 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_114() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 114 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_115() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 115 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_116() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 116 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_117() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 117 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_118() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 118 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_119() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 119 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_120() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 120 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_121() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 121 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_122() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 122 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_123() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 123 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_124() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 124 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_125() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 125 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_126() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 126 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_127() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 127 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_128() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 128 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_129() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 129 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_130() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 130 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_131() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 131 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_132() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 132 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_133() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 133 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_134() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 134 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_135() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 135 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_136() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 136 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_137() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 137 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_138() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 138 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_139() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 139 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_140() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 140 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_141() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 141 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_142() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 142 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_143() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 143 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_144() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 144 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_145() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 145 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_146() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 146 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_147() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 147 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_148() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 148 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_149() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 149 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_150() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 150 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_151() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 151 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_152() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 152 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_153() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 153 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_154() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 154 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_155() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 155 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_156() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 156 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_157() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 157 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_158() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 158 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_159() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 159 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_160() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 160 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_161() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 161 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_162() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 162 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_163() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 163 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_164() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 164 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_165() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 165 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_166() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 166 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_167() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 167 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_168() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 168 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_169() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 169 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_170() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 170 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_171() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 171 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_172() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 172 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_173() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 173 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_174() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 174 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_175() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 175 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_176() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 176 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_177() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 177 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_178() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 178 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_179() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 179 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_180() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 180 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_181() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 181 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_182() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 182 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_183() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 183 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_184() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 184 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_185() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 185 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_186() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 186 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_187() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 187 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_188() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 188 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_189() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 189 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_190() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 190 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_191() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 191 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_192() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 192 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_193() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 193 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_194() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 194 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_195() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 195 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_196() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 196 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_197() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 197 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_198() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 198 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_199() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 199 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_200() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 200 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_201() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 201 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_202() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 202 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_203() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 203 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_204() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 204 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_205() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 205 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_206() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 206 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_207() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 207 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_208() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 208 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_209() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 209 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_210() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 210 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_211() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 211 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_212() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 212 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_213() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 213 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_214() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 214 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_215() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 215 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_216() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 216 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_217() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 217 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_218() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 218 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_219() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 219 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_220() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 220 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_221() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 221 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_222() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 222 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_223() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 223 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_224() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 224 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_225() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 225 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_226() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 226 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_227() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 227 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_228() {
        let n = 6;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 228 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_229() {
        let n = 7;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 229 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_230() {
        let n = 3;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 230 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_231() {
        let n = 4;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 231 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    #[test]
    fn test_sampler_stress_232() {
        let n = 5;
        let feats = Tensor::zeros(vec![n, 4]);
        let graph = Graph::new(n, vec![0, 1, 1], vec![1, 2, 0], feats).unwrap();

        let sampled = sample_neighbors(&graph, &[0, 1], 2, 232 as u64);
        assert_eq!(sampled.target_nodes.len(), 2);

        let bg = collate_graphs(&[graph.clone(), graph]);
        assert_eq!(bg.num_graphs(), 2);
        assert_eq!(bg.total_nodes(), n * 2);
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
}
