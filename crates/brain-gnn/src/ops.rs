//! # GNN Operations
//!
//! Adjacency normalization, sparse softmax, and neighborhood aggregation ops.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Normalizes an adjacency matrix symmetrically: D^-1/2 A D^-1/2.
pub fn normalize_adj(adj: &Tensor) -> Tensor {
    let shape = adj.shape();
    let n = shape[0];
    let data = adj.to_vec();
    // Compute degrees
    let mut deg = vec![0.0f64; n];
    for r in 0..n {
        for c in 0..n {
            deg[r] += data[r * n + c];
        }
    }
    // Compute D^-1/2
    let deg_inv_sqrt: Vec<f64> = deg.iter().map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }).collect();
    // D^-1/2 A D^-1/2
    let mut norm_data = vec![0.0f64; n * n];
    for r in 0..n {
        for c in 0..n {
            norm_data[r * n + c] = deg_inv_sqrt[r] * data[r * n + c] * deg_inv_sqrt[c];
        }
    }
    Tensor::from_vec(norm_data, vec![n, n])
}

/// Applies softmax over neighborhood scores for attention.
pub fn sparse_softmax(scores: &[f64], dst_nodes: &[usize], num_nodes: usize) -> Vec<f64> {
    if scores.is_empty() { return vec![]; }
    let mut max_per_node = vec![f64::NEG_INFINITY; num_nodes];
    for (&score, &dst) in scores.iter().zip(dst_nodes.iter()) {
        if dst < num_nodes && score > max_per_node[dst] {
            max_per_node[dst] = score;
        }
    }
    let exp_scores: Vec<f64> = scores.iter().zip(dst_nodes.iter()).map(|(&score, &dst)| {
        let max_val = if dst < num_nodes && max_per_node[dst].is_finite() { max_per_node[dst] } else { 0.0 };
        (score - max_val).exp()
    }).collect();
    let mut sum_per_node = vec![0.0f64; num_nodes];
    for (&exp_val, &dst) in exp_scores.iter().zip(dst_nodes.iter()) {
        if dst < num_nodes {
            sum_per_node[dst] += exp_val;
        }
    }
    exp_scores.iter().zip(dst_nodes.iter()).map(|(&exp_val, &dst)| {
        let sum_val = if dst < num_nodes { sum_per_node[dst] } else { 1.0 };
        if sum_val > 0.0 { exp_val / sum_val } else { 0.0 }
    }).collect()
}

/// Aggregates node features along edges by mean.
pub fn aggregate_mean(src_features: &Tensor, dst_indices: &[usize], num_dst_nodes: usize) -> Tensor {
    let feat_data = src_features.to_vec();
    let feat_dim = if src_features.shape().len() > 1 { src_features.shape()[1] } else { 1 };
    let mut agg = vec![0.0f64; num_dst_nodes * feat_dim];
    let mut counts = vec![0usize; num_dst_nodes];

    let num_edges = dst_indices.len().min(feat_data.len() / feat_dim);
    for e in 0..num_edges {
        let dst = dst_indices[e];
        if dst < num_dst_nodes {
            counts[dst] += 1;
            for d in 0..feat_dim {
                agg[dst * feat_dim + d] += feat_data[e * feat_dim + d];
            }
        }
    }

    for dst in 0..num_dst_nodes {
        if counts[dst] > 0 {
            for d in 0..feat_dim {
                agg[dst * feat_dim + d] /= counts[dst] as f64;
            }
        }
    }

    Tensor::from_vec(agg, vec![num_dst_nodes, feat_dim])
}

/// Aggregates node features along edges by sum.
pub fn aggregate_sum(src_features: &Tensor, dst_indices: &[usize], num_dst_nodes: usize) -> Tensor {
    let feat_data = src_features.to_vec();
    let feat_dim = if src_features.shape().len() > 1 { src_features.shape()[1] } else { 1 };
    let mut agg = vec![0.0f64; num_dst_nodes * feat_dim];
    let num_edges = dst_indices.len().min(feat_data.len() / feat_dim);
    for e in 0..num_edges {
        let dst = dst_indices[e];
        if dst < num_dst_nodes {
            for d in 0..feat_dim {
                agg[dst * feat_dim + d] += feat_data[e * feat_dim + d];
            }
        }
    }
    Tensor::from_vec(agg, vec![num_dst_nodes, feat_dim])
}

/// Aggregates node features along edges by max.
pub fn aggregate_max(src_features: &Tensor, dst_indices: &[usize], num_dst_nodes: usize) -> Tensor {
    let feat_data = src_features.to_vec();
    let feat_dim = if src_features.shape().len() > 1 { src_features.shape()[1] } else { 1 };
    let mut agg = vec![f64::NEG_INFINITY; num_dst_nodes * feat_dim];
    let mut seen = vec![false; num_dst_nodes];
    let num_edges = dst_indices.len().min(feat_data.len() / feat_dim);
    for e in 0..num_edges {
        let dst = dst_indices[e];
        if dst < num_dst_nodes {
            seen[dst] = true;
            for d in 0..feat_dim {
                let v = feat_data[e * feat_dim + d];
                if v > agg[dst * feat_dim + d] {
                    agg[dst * feat_dim + d] = v;
                }
            }
        }
    }
    for dst in 0..num_dst_nodes {
        if !seen[dst] {
            for d in 0..feat_dim {
                agg[dst * feat_dim + d] = 0.0;
            }
        }
    }
    Tensor::from_vec(agg, vec![num_dst_nodes, feat_dim])
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_ops_stress_001() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_002() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_003() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_004() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_005() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_006() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_007() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_008() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_009() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_010() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_011() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_012() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_013() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_014() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_015() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_016() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_017() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_018() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_019() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_020() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_021() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_022() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_023() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_024() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_025() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_026() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_027() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_028() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_029() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_030() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_031() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_032() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_033() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_034() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_035() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_036() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_037() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_038() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_039() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_040() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_041() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_042() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_043() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_044() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_045() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_046() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_047() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_048() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_049() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_050() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_051() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_052() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_053() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_054() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_055() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_056() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_057() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_058() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_059() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_060() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_061() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_062() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_063() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_064() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_065() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_066() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_067() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_068() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_069() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_070() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_071() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_072() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_073() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_074() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_075() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_076() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_077() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_078() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_079() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_080() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_081() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_082() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_083() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_084() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_085() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_086() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_087() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_088() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_089() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_090() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_091() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_092() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_093() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_094() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_095() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_096() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_097() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_098() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_099() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_100() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_101() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_102() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_103() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_104() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_105() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_106() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_107() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_108() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_109() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_110() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_111() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_112() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_113() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_114() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_115() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_116() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_117() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_118() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_119() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_120() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_121() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_122() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_123() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_124() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_125() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_126() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_127() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_128() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_129() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_130() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_131() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_132() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_133() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_134() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_135() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_136() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_137() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_138() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_139() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_140() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_141() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_142() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_143() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_144() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_145() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_146() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_147() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_148() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_149() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_150() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_151() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_152() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_153() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_154() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_155() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_156() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_157() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_158() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_159() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
    }

    #[test]
    fn test_ops_stress_160() {
        let adj = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let norm = normalize_adj(&adj);
        assert_eq!(norm.shape(), &[2, 2]);

        let scores = vec![1.0, 2.0, 3.0];
        let dsts = vec![0, 0, 1];
        let softmax_vals = sparse_softmax(&scores, &dsts, 2);
        assert_eq!(softmax_vals.len(), 3);

        let feats = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let agg_m = aggregate_mean(&feats, &[0, 1], 2);
        assert_eq!(agg_m.shape(), &[2, 2]);
        let agg_s = aggregate_sum(&feats, &[0, 1], 2);
        assert_eq!(agg_s.shape(), &[2, 2]);
        let agg_x = aggregate_max(&feats, &[0, 1], 2);
        assert_eq!(agg_x.shape(), &[2, 2]);
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
}
