//! # Readout Operations
//!
//! Pooling functions: global_add_pool, global_mean_pool, global_max_pool, global_attention_pool, set2set.
#![allow(missing_docs)]

pub mod jumping;
pub use jumping::{JumpingKnowledge, JkConfig, JkMode};

use brain_core::Tensor;

/// Sum pooling over graph nodes per graph in batch.
pub fn global_add_pool(x: &Tensor, batch_indices: &[usize], num_graphs: usize) -> Tensor {
    let feat_dim = if x.shape().len() > 1 { x.shape()[1] } else { 1 };
    let data = x.to_vec();
    let num_nodes = batch_indices.len().min(x.shape()[0]);

    let mut pool = vec![0.0f64; num_graphs * feat_dim];
    for n in 0..num_nodes {
        let g = batch_indices[n];
        if g < num_graphs {
            for d in 0..feat_dim {
                pool[g * feat_dim + d] += data[n * feat_dim + d];
            }
        }
    }

    Tensor::from_vec(pool, vec![num_graphs, feat_dim])
}

/// Mean pooling over graph nodes per graph in batch.
pub fn global_mean_pool(x: &Tensor, batch_indices: &[usize], num_graphs: usize) -> Tensor {
    let feat_dim = if x.shape().len() > 1 { x.shape()[1] } else { 1 };
    let data = x.to_vec();
    let num_nodes = batch_indices.len().min(x.shape()[0]);

    let mut pool = vec![0.0f64; num_graphs * feat_dim];
    let mut counts = vec![0usize; num_graphs];

    for n in 0..num_nodes {
        let g = batch_indices[n];
        if g < num_graphs {
            counts[g] += 1;
            for d in 0..feat_dim {
                pool[g * feat_dim + d] += data[n * feat_dim + d];
            }
        }
    }

    for g in 0..num_graphs {
        if counts[g] > 0 {
            for d in 0..feat_dim {
                pool[g * feat_dim + d] /= counts[g] as f64;
            }
        }
    }

    Tensor::from_vec(pool, vec![num_graphs, feat_dim])
}

/// Max pooling over graph nodes per graph in batch.
pub fn global_max_pool(x: &Tensor, batch_indices: &[usize], num_graphs: usize) -> Tensor {
    let feat_dim = if x.shape().len() > 1 { x.shape()[1] } else { 1 };
    let data = x.to_vec();
    let num_nodes = batch_indices.len().min(x.shape()[0]);

    let mut pool = vec![f64::NEG_INFINITY; num_graphs * feat_dim];
    let mut seen = vec![false; num_graphs];

    for n in 0..num_nodes {
        let g = batch_indices[n];
        if g < num_graphs {
            seen[g] = true;
            for d in 0..feat_dim {
                let v = data[n * feat_dim + d];
                if v > pool[g * feat_dim + d] {
                    pool[g * feat_dim + d] = v;
                }
            }
        }
    }

    for g in 0..num_graphs {
        if !seen[g] {
            for d in 0..feat_dim {
                pool[g * feat_dim + d] = 0.0;
            }
        }
    }

    Tensor::from_vec(pool, vec![num_graphs, feat_dim])
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant)]
    use super::*;
    use brain_core::Tensor;
}
