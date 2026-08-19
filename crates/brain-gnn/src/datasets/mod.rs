//! # Synthetic Graph Datasets
//!
//! Erdos-Renyi with planted communities, CycleDataset, Zachary's Karate Club.
#![allow(missing_docs)]

pub mod loader;
pub use loader::{GraphBatch, GraphLoader};

use brain_core::Tensor;
use crate::graph::Graph;

/// Split mask for dataset partitioning.
#[derive(Debug, Clone)]
pub struct DatasetSplits {
    pub train_mask: Vec<bool>,
    pub val_mask: Vec<bool>,
    pub test_mask: Vec<bool>,
}

impl DatasetSplits {
    pub fn new(num_nodes: usize, train_ratio: f64, val_ratio: f64) -> Self {
        let n_train = (num_nodes as f64 * train_ratio).floor() as usize;
        let n_val = (num_nodes as f64 * val_ratio).floor() as usize;

        let mut train_mask = vec![false; num_nodes];
        let mut val_mask = vec![false; num_nodes];
        let mut test_mask = vec![false; num_nodes];

        for i in 0..num_nodes {
            if i < n_train {
                train_mask[i] = true;
            } else if i < n_train + n_val {
                val_mask[i] = true;
            } else {
                test_mask[i] = true;
            }
        }

        Self { train_mask, val_mask, test_mask }
    }
}

/// Random graph dataset with planted communities.
pub fn random_community_graph(num_nodes: usize, num_communities: usize, seed: u64) -> (Graph, Vec<usize>) {
    let mut rng = seed;
    let lcg = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    };

    let labels: Vec<usize> = (0..num_nodes).map(|i| i % num_communities.max(1)).collect();
    let mut src = Vec::new();
    let mut dst = Vec::new();

    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if i != j {
                let p = if labels[i] == labels[j] { 0.4 } else { 0.05 };
                if lcg(&mut rng) < p {
                    src.push(i);
                    dst.push(j);
                }
            }
        }
    }

    let feat_dim = 16;
    let mut feats = vec![0.0f64; num_nodes * feat_dim];
    for i in 0..num_nodes {
        let c = labels[i];
        for d in 0..feat_dim {
            feats[i * feat_dim + d] = if d == c % feat_dim { 1.0 } else { 0.1 * lcg(&mut rng) };
        }
    }

    let g = Graph::new(num_nodes, src, dst, Tensor::from_vec(feats, vec![num_nodes, feat_dim])).unwrap();
    (g, labels)
}

/// Cycle graph dataset of N nodes.
pub fn cycle_graph(num_nodes: usize) -> Graph {
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for i in 0..num_nodes {
        let next = (i + 1) % num_nodes;
        src.push(i);
        dst.push(next);
        src.push(next);
        dst.push(i);
    }
    let feats = Tensor::zeros(vec![num_nodes, 4]);
    Graph::new(num_nodes, src, dst, feats).unwrap()
}

/// Zachary's Karate Club 34-node benchmark graph.
pub fn zachary_karate_club() -> (Graph, Vec<usize>) {
    let num_nodes = 34;
    let mut src = Vec::new();
    let mut dst = Vec::new();

    let edges = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
        (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
        (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
        (3, 7), (3, 12), (3, 13),
        (4, 6), (4, 10),
        (5, 6), (5, 10), (5, 16),
        (6, 16),
        (8, 30), (8, 32), (8, 33),
        (9, 33),
        (13, 33),
        (14, 32), (14, 33),
        (15, 32), (15, 33),
        (18, 32), (18, 33),
        (19, 33),
        (20, 32), (20, 33),
        (22, 32), (22, 33),
        (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
        (24, 25), (24, 27), (24, 31),
        (25, 31),
        (26, 29), (26, 33),
        (27, 33),
        (28, 31), (28, 33),
        (29, 32), (29, 33),
        (30, 32), (30, 33),
        (31, 32), (31, 33),
        (32, 33),
    ];

    for &(u, v) in &edges {
        src.push(u);
        dst.push(v);
        src.push(v);
        dst.push(u);
    }

    let labels: Vec<usize> = (0..34).map(|i| if i < 17 { 0 } else { 1 }).collect();
    let feats = Tensor::from_vec(vec![1.0; 34 * 4], vec![34, 4]);
    let graph = Graph::new(num_nodes, src, dst, feats).unwrap();
    (graph, labels)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
