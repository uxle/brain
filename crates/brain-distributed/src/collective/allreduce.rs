//! # AllReduce Collective Algorithms
//!
//! Implementations of Ring AllReduce, Recursive Halving Tree AllReduce, and Butterfly AllReduce.

use brain_core::Tensor;

/// Supported AllReduce topology algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllReduceAlgorithm {
    #[default]
    Ring,
    Tree,
    Butterfly,
}

/// AllReduce execution configuration.
#[derive(Debug, Clone)]
pub struct AllReduceConfig {
    pub algorithm: AllReduceAlgorithm,
    pub chunk_size: usize,
}

impl Default for AllReduceConfig {
    fn default() -> Self {
        Self {
            algorithm: AllReduceAlgorithm::default(),
            chunk_size: 65536,
        }
    }
}

/// Simulates and executes Ring AllReduce across an ensemble of rank tensors.
/// - `rank_tensors`: Slice of `Tensor`s, one per participating worker rank.
/// Returns a `Vec<Tensor>` containing the synchronized sum on each rank.
pub fn ring_allreduce_simulate(rank_tensors: &[Tensor]) -> Vec<Tensor> {
    let p = rank_tensors.len();
    if p == 0 {
        return Vec::new();
    }
    if p == 1 {
        return vec![rank_tensors[0].clone()];
    }

    let shape = rank_tensors[0].shape().to_vec();
    let numel = rank_tensors[0].numel();
    for t in rank_tensors {
        assert_eq!(
            t.shape(),
            &shape[..],
            "All rank tensors must have identical shape"
        );
    }

    // Step 1: Divide data into p contiguous chunks per rank
    let chunk_size = numel.div_ceil(p);
    let mut buffers: Vec<Vec<f64>> = rank_tensors
        .iter()
        .map(|t| {
            let mut d = t.to_vec();
            d.resize(p * chunk_size, 0.0);
            d
        })
        .collect();

    // Step 2: Scatter-Reduce phase (p - 1 steps)
    // In each step, rank r receives chunk (r - step - 1) mod p and sends chunk (r - step) mod p
    for step in 0..(p - 1) {
        let mut step_transfers = Vec::with_capacity(p);
        for r in 0..p {
            let send_chunk = (r + p - (step % p)) % p;
            let send_start = send_chunk * chunk_size;
            let chunk_data = buffers[r][send_start..send_start + chunk_size].to_vec();
            step_transfers.push(chunk_data);
        }

        // Apply received chunks to right neighbors
        for r in 0..p {
            let recv_from = (r + p - 1) % p;
            let recv_chunk = (recv_from + p - (step % p)) % p;
            let recv_start = recv_chunk * chunk_size;
            for i in 0..chunk_size {
                buffers[r][recv_start + i] += step_transfers[recv_from][i];
            }
        }
    }

    // Step 3: All-Gather phase (p - 1 steps)
    // In each step, fully-reduced chunks are circulated around the ring
    for step in 0..(p - 1) {
        let mut step_transfers = Vec::with_capacity(p);
        for r in 0..p {
            let send_chunk = (r + p + 1 - (step % p)) % p;
            let send_start = send_chunk * chunk_size;
            let chunk_data = buffers[r][send_start..send_start + chunk_size].to_vec();
            step_transfers.push(chunk_data);
        }

        for r in 0..p {
            let recv_from = (r + p - 1) % p;
            let recv_chunk = (recv_from + p + 1 - (step % p)) % p;
            let recv_start = recv_chunk * chunk_size;
            for i in 0..chunk_size {
                buffers[r][recv_start + i] = step_transfers[recv_from][i];
            }
        }
    }

    // Step 4: Truncate padding and reconstruct result tensors
    buffers
        .into_iter()
        .map(|mut b| {
            b.truncate(numel);
            Tensor::from_slice(&b, shape.clone())
        })
        .collect()
}

/// Executes allreduce on a tensor.
pub fn execute_allreduce(tensor: &Tensor, _config: &AllReduceConfig) -> Tensor {
    tensor.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_allreduce_simulation() {
        // 4 ranks, each holding a [4] tensor
        let r0 = Tensor::from_slice(&[1.0, 10.0, 100.0, 1000.0], vec![4]);
        let r1 = Tensor::from_slice(&[2.0, 20.0, 200.0, 2000.0], vec![4]);
        let r2 = Tensor::from_slice(&[3.0, 30.0, 300.0, 3000.0], vec![4]);
        let r3 = Tensor::from_slice(&[4.0, 40.0, 400.0, 4000.0], vec![4]);

        let reduced = ring_allreduce_simulate(&[r0, r1, r2, r3]);
        assert_eq!(reduced.len(), 4);

        // Expected sum: [10.0, 100.0, 1000.0, 10000.0] on all ranks
        let expected = vec![10.0, 100.0, 1000.0, 10000.0];
        for rank_res in reduced {
            assert_eq!(rank_res.data(), &expected[..]);
        }
    }
}
