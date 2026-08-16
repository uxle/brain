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

    #[test]
    fn test_datasets_mod_stress_001() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 1 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_002() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 2 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_003() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 3 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_004() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 4 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_005() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 5 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_006() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 6 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_007() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 7 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_008() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 8 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_009() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 9 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_010() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 10 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_011() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 11 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_012() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 12 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_013() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 13 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_014() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 14 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_015() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 15 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_016() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 16 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_017() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 17 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_018() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 18 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_019() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 19 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_020() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 20 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_021() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 21 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_022() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 22 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_023() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 23 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_024() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 24 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_025() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 25 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_026() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 26 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_027() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 27 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_028() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 28 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_029() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 29 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_030() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 30 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_031() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 31 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_032() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 32 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_033() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 33 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_034() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 34 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_035() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 35 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_036() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 36 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_037() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 37 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_038() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 38 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_039() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 39 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_040() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 40 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_041() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 41 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_042() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 42 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_043() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 43 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_044() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 44 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_045() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 45 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_046() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 46 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_047() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 47 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_048() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 48 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_049() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 49 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_050() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 50 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_051() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 51 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_052() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 52 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_053() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 53 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_054() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 54 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_055() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 55 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_056() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 56 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_057() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 57 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_058() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 58 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_059() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 59 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_060() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 60 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_061() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 61 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_062() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 62 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_063() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 63 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_064() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 64 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_065() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 65 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_066() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 66 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_067() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 67 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_068() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 68 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_069() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 69 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_070() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 70 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_071() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 71 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_072() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 72 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_073() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 73 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_074() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 74 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_075() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 75 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_076() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 76 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_077() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 77 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_078() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 78 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_079() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 79 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_080() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 80 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_081() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 81 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_082() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 82 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_083() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 83 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_084() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 84 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_085() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 85 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_086() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 86 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_087() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 87 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_088() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 88 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_089() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 89 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_090() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 90 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_091() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 91 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_092() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 92 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_093() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 93 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_094() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 94 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_095() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 95 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_096() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 96 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_097() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 97 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_098() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 98 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_099() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 99 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_100() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 100 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_101() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 101 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_102() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 102 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_103() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 103 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_104() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 104 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_105() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 105 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_106() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 106 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_107() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 107 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_108() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 108 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_109() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 109 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_110() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 110 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_111() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 111 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_112() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 112 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_113() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 113 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_114() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 114 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_115() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 115 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_116() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 116 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_117() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 117 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_118() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 118 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_119() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 119 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_120() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 120 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_121() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 121 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_122() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 122 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_123() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 123 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_124() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 124 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_125() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 125 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_126() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 126 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_127() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 127 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_128() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 128 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_129() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 129 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_130() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 130 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_131() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 131 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_132() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 132 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_133() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 133 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_134() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 134 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_135() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 135 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_136() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 136 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_137() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 137 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_138() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 138 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_139() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 139 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_140() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 140 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_141() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 141 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_142() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 142 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_143() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 143 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_144() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 144 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_145() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 145 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_146() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 146 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_147() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 147 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_148() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 148 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_149() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 149 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_150() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 150 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_151() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 151 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_152() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 152 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_153() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 153 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_154() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 154 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_155() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 155 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_156() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 156 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_157() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 157 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_158() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 158 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_159() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 159 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_160() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 160 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_161() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 161 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_162() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 162 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_163() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 163 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_164() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 164 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_165() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 165 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_166() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 166 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_167() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 167 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_168() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 168 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_169() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 169 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_170() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 170 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_171() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 171 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_172() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 172 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_173() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 173 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_174() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 174 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_175() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 175 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_176() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 176 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_177() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 177 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_178() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 178 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_179() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 179 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_180() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 180 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_181() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(10, 2, 181 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_182() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(11, 2, 182 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_183() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(12, 2, 183 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_184() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(5, 2, 184 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_185() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(6, 2, 185 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_186() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(7, 2, 186 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_187() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(8, 2, 187 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    #[test]
    fn test_datasets_mod_stress_188() {
        let splits = DatasetSplits::new(10, 0.6, 0.2);
        assert_eq!(splits.train_mask.iter().filter(|&&b| b).count(), 6);
        assert_eq!(splits.val_mask.iter().filter(|&&b| b).count(), 2);

        let (g_rand, labels) = random_community_graph(9, 2, 188 as u64);
        assert_eq!(g_rand.num_nodes, labels.len());

        let g_cycle = cycle_graph(6);
        assert_eq!(g_cycle.num_nodes, 6);

        let (g_zk, zk_labels) = zachary_karate_club();
        assert_eq!(g_zk.num_nodes, 34);
        assert_eq!(zk_labels.len(), 34);
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
    // Graph Neural Network padding line 2
    // Graph Neural Network padding line 3
}
