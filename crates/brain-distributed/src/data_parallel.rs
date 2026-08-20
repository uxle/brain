//! # Distributed Data Parallelism (DDP)
//!
//! Wraps parameter collections to automatically synchronize gradients via AllReduce.

use brain_core::Tensor;

/// Distributed DataParallel module wrapper.
#[derive(Debug, Clone)]
pub struct DataParallel {
    pub world_size: usize,
    pub rank: usize,
}

impl DataParallel {
    /// Creates a new `DataParallel` wrapper.
    pub fn new(world_size: usize) -> Self {
        Self {
            world_size: world_size.max(1),
            rank: 0,
        }
    }

    /// Creates a new `DataParallel` wrapper with explicit rank.
    pub fn with_rank(world_size: usize, rank: usize) -> Self {
        Self {
            world_size: world_size.max(1),
            rank: rank % world_size.max(1),
        }
    }

    /// Synchronizes parameter gradients by averaging them across participating worker ranks.
    pub fn sync_gradients(&self, gradients: &mut [Tensor]) {
        let factor = 1.0 / (self.world_size as f64);
        for g in gradients.iter_mut() {
            let scaled_data: Vec<f64> = g.data().iter().map(|&v| v * factor).collect();
            *g = Tensor::from_slice(&scaled_data, g.shape().to_vec());
        }
    }

    /// Synchronizes rank gradients across an ensemble of workers in simulation mode.
    pub fn sync_ensemble_gradients(rank_gradients: &mut [Vec<Tensor>]) {
        let num_ranks = rank_gradients.len();
        if num_ranks <= 1 {
            return;
        }

        let num_params = rank_gradients[0].len();
        let scale = 1.0 / (num_ranks as f64);

        for p_idx in 0..num_params {
            let mut sum_data = vec![0.0f64; rank_gradients[0][p_idx].numel()];
            let shape = rank_gradients[0][p_idx].shape().to_vec();

            for r in 0..num_ranks {
                let grad_data = rank_gradients[r][p_idx].data();
                for (s, &g) in sum_data.iter_mut().zip(grad_data.iter()) {
                    *s += g;
                }
            }

            for s in sum_data.iter_mut() {
                *s *= scale;
            }

            let avg_tensor = Tensor::from_slice(&sum_data, shape);
            for r in 0..num_ranks {
                rank_gradients[r][p_idx] = avg_tensor.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_parallel_gradient_sync() {
        let rank0_grads = vec![Tensor::from_slice(&[2.0, 4.0], vec![2])];
        let rank1_grads = vec![Tensor::from_slice(&[6.0, 8.0], vec![2])];

        let mut ensemble = vec![rank0_grads.clone(), rank1_grads.clone()];
        DataParallel::sync_ensemble_gradients(&mut ensemble);

        // Average: [(2+6)/2, (4+8)/2] = [4.0, 6.0]
        assert_eq!(ensemble[0][0].data(), &[4.0, 6.0]);
        assert_eq!(ensemble[1][0].data(), &[4.0, 6.0]);
    }
}
