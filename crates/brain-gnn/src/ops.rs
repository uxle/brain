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
}
