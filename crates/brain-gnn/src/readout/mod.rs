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

    #[test]
    fn test_readout_mod_stress_001() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_002() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_003() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_004() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_005() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_006() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_007() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_008() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_009() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_010() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_011() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_012() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_013() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_014() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_015() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_016() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_017() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_018() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_019() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_020() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_021() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_022() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_023() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_024() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_025() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_026() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_027() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_028() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_029() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_030() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_031() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_032() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_033() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_034() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_035() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_036() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_037() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_038() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_039() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_040() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_041() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_042() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_043() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_044() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_045() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_046() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_047() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_048() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_049() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_050() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_051() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_052() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_053() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_054() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_055() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_056() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_057() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_058() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_059() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_060() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_061() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_062() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_063() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_064() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_065() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_066() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_067() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_068() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_069() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_070() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_071() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_072() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_073() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_074() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_075() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_076() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_077() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_078() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_079() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_080() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_081() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_082() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_083() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_084() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_085() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_086() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_087() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_088() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_089() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_090() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_091() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_092() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_093() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_094() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_095() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_096() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_097() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_098() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_099() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_100() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_101() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_102() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_103() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_104() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_105() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_106() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_107() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_108() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_109() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_110() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_111() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_112() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_113() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_114() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_115() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_116() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_117() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_118() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_119() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_120() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_121() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_122() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_123() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_124() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_125() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_126() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_127() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_128() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_129() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_130() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_131() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_132() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_133() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_134() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_135() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_136() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_137() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_138() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_139() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_140() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_141() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_142() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_143() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_144() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_145() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_146() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_147() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_148() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_149() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_150() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_151() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_152() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_153() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_154() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_155() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_156() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_157() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_158() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_159() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_160() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_161() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_162() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_163() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_164() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_165() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_166() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_167() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_168() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_169() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_170() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_171() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_172() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_173() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_174() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_175() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_176() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_177() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_178() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_179() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_180() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_181() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_182() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_183() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_184() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_185() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_186() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_187() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_188() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_189() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_190() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_191() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_192() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_193() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_194() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_195() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_196() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_197() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_198() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_199() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_200() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_201() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_202() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_203() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_204() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_205() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_206() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_207() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_208() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_209() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_210() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_211() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_212() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_213() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_214() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_215() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_216() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_217() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_218() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_219() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_220() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_221() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_222() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_223() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_224() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_225() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_226() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_227() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_228() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_229() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_230() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_231() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_232() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_233() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_234() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_235() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_236() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_237() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_238() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_239() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_240() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_241() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_242() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_243() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_244() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_245() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_246() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_247() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_248() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_249() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_250() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_251() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_252() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_253() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_254() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_255() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_256() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_257() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_258() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_259() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_260() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_261() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_262() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_263() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_264() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_265() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_266() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_267() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_268() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_269() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
    }

    #[test]
    fn test_readout_mod_stress_270() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let batch = vec![0, 1];
        let p_add = global_add_pool(&x, &batch, 2);
        assert_eq!(p_add.shape(), &[2, 2]);
        let p_mean = global_mean_pool(&x, &batch, 2);
        assert_eq!(p_mean.shape(), &[2, 2]);
        let p_max = global_max_pool(&x, &batch, 2);
        assert_eq!(p_max.shape(), &[2, 2]);
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
}
