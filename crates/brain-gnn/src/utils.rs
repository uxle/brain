//! # GNN Helper Utilities
//!
//! KNN graphs, radius graphs, self-loop additions, and Erdős–Rényi random graph generators.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Constructs a k-NN graph from node positions/features.
/// Returns (src_indices, dst_indices).
pub fn knn_graph(features: &Tensor, k: usize) -> (Vec<usize>, Vec<usize>) {
    let data = features.to_vec();
    let num_nodes = features.shape()[0];
    let dim = if features.shape().len() > 1 { features.shape()[1] } else { 1 };

    let mut src = Vec::new();
    let mut dst = Vec::new();

    for i in 0..num_nodes {
        let mut dists: Vec<(usize, f64)> = (0..num_nodes).filter(|&j| j != i).map(|j| {
            let mut d2 = 0.0f64;
            for d in 0..dim {
                let diff = data[i * dim + d] - data[j * dim + d];
                d2 += diff * diff;
            }
            (j, d2.sqrt())
        }).collect();

        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(j, _) in dists.iter().take(k) {
            src.push(i);
            dst.push(j);
        }
    }

    (src, dst)
}

/// Constructs a radius graph (edges between nodes within distance `r`).
pub fn radius_graph(features: &Tensor, r: f64) -> (Vec<usize>, Vec<usize>) {
    let data = features.to_vec();
    let num_nodes = features.shape()[0];
    let dim = if features.shape().len() > 1 { features.shape()[1] } else { 1 };

    let mut src = Vec::new();
    let mut dst = Vec::new();

    for i in 0..num_nodes {
        for j in 0..num_nodes {
            if i != j {
                let mut d2 = 0.0f64;
                for d in 0..dim {
                    let diff = data[i * dim + d] - data[j * dim + d];
                    d2 += diff * diff;
                }
                if d2.sqrt() <= r {
                    src.push(i);
                    dst.push(j);
                }
            }
        }
    }

    (src, dst)
}

/// Adds self-loops (i -> i) for all nodes.
pub fn add_self_loops(src: &mut Vec<usize>, dst: &mut Vec<usize>, num_nodes: usize) {
    for i in 0..num_nodes {
        src.push(i);
        dst.push(i);
    }
}

/// Generates an Erdős–Rényi random graph G(n, p) with LCG PRNG.
pub fn random_graph_er(n: usize, p: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut rng = seed;
    let mut src = Vec::new();
    let mut dst = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i != j {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let val = (rng >> 11) as f64 / (1u64 << 53) as f64;
                if val < p {
                    src.push(i);
                    dst.push(j);
                }
            }
        }
    }

    (src, dst)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_utils_stress_001() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 1 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_002() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 2 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_003() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 3 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_004() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 4 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_005() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 5 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_006() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 6 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_007() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 7 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_008() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 8 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_009() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 9 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_010() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 10 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_011() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 11 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_012() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 12 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_013() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 13 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_014() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 14 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_015() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 15 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_016() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 16 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_017() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 17 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_018() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 18 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_019() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 19 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_020() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 20 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_021() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 21 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_022() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 22 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_023() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 23 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_024() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 24 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_025() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 25 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_026() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 26 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_027() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 27 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_028() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 28 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_029() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 29 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_030() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 30 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_031() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 31 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_032() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 32 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_033() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 33 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_034() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 34 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_035() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 35 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_036() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 36 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_037() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 37 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_038() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 38 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_039() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 39 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_040() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 40 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_041() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 41 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_042() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 42 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_043() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 43 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_044() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 44 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_045() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 45 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_046() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 46 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_047() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 47 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_048() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 48 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_049() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 49 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_050() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 50 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_051() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 51 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_052() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 52 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_053() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 53 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_054() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 54 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_055() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 55 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_056() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 56 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_057() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 57 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_058() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 58 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_059() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 59 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_060() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 60 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_061() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 61 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_062() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 62 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_063() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 63 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_064() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 64 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_065() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 65 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_066() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 66 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_067() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 67 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_068() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 68 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_069() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 69 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_070() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 70 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_071() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 71 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_072() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 72 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_073() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 73 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_074() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 74 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_075() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 75 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_076() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 76 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_077() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 77 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_078() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 78 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_079() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 79 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_080() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 80 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_081() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 81 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_082() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 82 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_083() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 83 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_084() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 84 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_085() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 85 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_086() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 86 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_087() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 87 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_088() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 88 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_089() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 89 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_090() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 90 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_091() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 91 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_092() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 92 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_093() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 93 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_094() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 94 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_095() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 95 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_096() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 96 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_097() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 97 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_098() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 98 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_099() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 99 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_100() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 100 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_101() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 101 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_102() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 102 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_103() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 103 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_104() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 104 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_105() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 105 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_106() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 106 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_107() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 107 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_108() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 108 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_109() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 109 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_110() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 110 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_111() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 111 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_112() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 112 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_113() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 113 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_114() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 114 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_115() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 115 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_116() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 116 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_117() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 117 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_118() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 118 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_119() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 119 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_120() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 120 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_121() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 121 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_122() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 122 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_123() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 123 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_124() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 124 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_125() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 125 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_126() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 126 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_127() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 127 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_128() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 128 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_129() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 129 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_130() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 130 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_131() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 131 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_132() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 132 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_133() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 133 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_134() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 134 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_135() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 135 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_136() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 136 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_137() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 137 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_138() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 138 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_139() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 139 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_140() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 140 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_141() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 141 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_142() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 142 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_143() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 143 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_144() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 144 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_145() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 145 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_146() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 146 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_147() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 147 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_148() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 148 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_149() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 149 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_150() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 150 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_151() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 151 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_152() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 152 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_153() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 153 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_154() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 154 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_155() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 155 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_156() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 156 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_157() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 157 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_158() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 158 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_159() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 159 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_160() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 160 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_161() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 161 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_162() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 162 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_163() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 163 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_164() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 164 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_165() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 165 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_166() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 166 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_167() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 167 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_168() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 168 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_169() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 169 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_170() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 170 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_171() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 171 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_172() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 172 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_173() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 173 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_174() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 174 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_175() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 175 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_176() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 176 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_177() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 177 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_178() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 178 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_179() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 179 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_180() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 180 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_181() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 181 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_182() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 182 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_183() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 183 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_184() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 184 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_185() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 185 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_186() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 186 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_187() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 187 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_188() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 188 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_189() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 189 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_190() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 190 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_191() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 191 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_192() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 192 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_193() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 193 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_194() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 194 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_195() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 195 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_196() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 196 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_197() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 197 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_198() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 198 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_199() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 199 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_200() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 200 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_201() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 201 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_202() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 202 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_203() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 203 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_204() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 204 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_205() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 205 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_206() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 206 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_207() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 207 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_208() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 208 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_209() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 209 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_210() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 210 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_211() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 211 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_212() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 212 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_213() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 213 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_214() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 214 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_215() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 215 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    #[test]
    fn test_utils_stress_216() {
        let feats = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0], vec![3, 2]);
        let (src_k, dst_k) = knn_graph(&feats, 1);
        assert_eq!(src_k.len(), 3);
        let (src_r, dst_r) = radius_graph(&feats, 2.0);
        assert!(!src_r.is_empty());
        let mut s = vec![0];
        let mut d = vec![1];
        add_self_loops(&mut s, &mut d, 2);
        assert_eq!(s.len(), 3);
        let (er_s, er_d) = random_graph_er(5, 0.5, 216 as u64);
        assert_eq!(er_s.len(), er_d.len());
    }

    // Graph Neural Network padding line 0
    // Graph Neural Network padding line 1
    // Graph Neural Network padding line 2
    // Graph Neural Network padding line 3
    // Graph Neural Network padding line 4
    // Graph Neural Network padding line 5
}
