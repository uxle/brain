//! # Secure Aggregation (Bonawitz et al. / SecAgg)
//!
//! Mask-based secure aggregation using zero-sum pairwise random masks to hide individual client updates.

use brain_core::Tensor;

/// Secure aggregation coordinator using shared pseudo-random masks.
#[derive(Debug, Clone)]
pub struct SecureAggregator {
    pub num_clients: usize,
}

impl SecureAggregator {
    /// Creates a new `SecureAggregator`.
    pub fn new(num_clients: usize) -> Self {
        Self { num_clients }
    }

    /// Generates the net zero-sum mask for client `i` across all pairs `(i, j)`.
    /// For every pair (i, j) with i < j, client i adds mask M_{i,j} and client j subtracts M_{i,j}.
    /// The sum over all client masks cancels out to exactly 0.0!
    pub fn generate_client_pairwise_mask(
        &self,
        client_id: usize,
        shape: &[usize],
        seed: u64,
    ) -> Tensor {
        let mut total_mask = Tensor::zeros(shape.to_vec());

        for peer_id in 0..self.num_clients {
            if peer_id == client_id {
                continue;
            }
            let (min_id, max_id) = if client_id < peer_id {
                (client_id, peer_id)
            } else {
                (peer_id, client_id)
            };

            let pair_seed = seed
                .wrapping_add((min_id as u64).wrapping_mul(0x9e3779b97f4a7c15))
                .wrapping_add((max_id as u64).wrapping_mul(0x517cc1b727220a95));

            let pair_mask = generate_mask(shape.to_vec(), min_id * 1000 + max_id, pair_seed);

            if client_id < peer_id {
                total_mask = &total_mask + &pair_mask;
            } else {
                total_mask = &total_mask - &pair_mask;
            }
        }

        total_mask
    }
}

/// Generates a pseudo-random mask tensor for a given client and round seed.
pub fn generate_mask(shape: Vec<usize>, client_id: usize, round_seed: u64) -> Tensor {
    let n: usize = shape.iter().product();
    let mut rng = round_seed.wrapping_add((client_id as u64).wrapping_mul(0x9e3779b97f4a7c15));
    let data: Vec<f64> = (0..n)
        .map(|_| {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng as i64) as f64) / (i64::MAX as f64)
        })
        .collect();
    Tensor::from_vec(data, shape)
}

/// Applies a mask to a tensor.
pub fn mask_tensor(t: &Tensor, mask: &Tensor) -> Tensor {
    t + mask
}

/// Removes a mask from a tensor.
pub fn unmask_tensor(t: &Tensor, mask: &Tensor) -> Tensor {
    t - mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_aggregation_zero_sum_cancellation() {
        let n_clients = 4;
        let sec_agg = SecureAggregator::new(n_clients);
        let shape = vec![2, 2];
        let seed = 42;

        let client_weights = vec![
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], shape.clone()),
            Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], shape.clone()),
            Tensor::from_slice(&[100.0, 200.0, 300.0, 400.0], shape.clone()),
            Tensor::from_slice(&[1000.0, 2000.0, 3000.0, 4000.0], shape.clone()),
        ];

        // Mask each client update
        let mut masked_updates = Vec::new();
        for i in 0..n_clients {
            let mask = sec_agg.generate_client_pairwise_mask(i, &shape, seed);
            masked_updates.push(mask_tensor(&client_weights[i], &mask));
        }

        // Server sums all masked updates
        let mut server_sum = Tensor::zeros(shape.clone());
        for m in &masked_updates {
            server_sum = &server_sum + m;
        }

        // True unmasked sum
        let mut true_sum = Tensor::zeros(shape);
        for w in &client_weights {
            true_sum = &true_sum + w;
        }

        // All pairwise masks cancelled out to exact 0.0!
        for (s, t) in server_sum.data().iter().zip(true_sum.data().iter()) {
            assert!((s - t).abs() < 1e-6);
        }
    }
}
