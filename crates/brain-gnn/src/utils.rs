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
    let dim = if features.shape().len() > 1 {
        features.shape()[1]
    } else {
        1
    };

    let mut src = Vec::new();
    let mut dst = Vec::new();

    for i in 0..num_nodes {
        let mut dists: Vec<(usize, f64)> = (0..num_nodes)
            .filter(|&j| j != i)
            .map(|j| {
                let mut d2 = 0.0f64;
                for d in 0..dim {
                    let diff = data[i * dim + d] - data[j * dim + d];
                    d2 += diff * diff;
                }
                (j, d2.sqrt())
            })
            .collect();

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
    let dim = if features.shape().len() > 1 {
        features.shape()[1]
    } else {
        1
    };

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
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
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
