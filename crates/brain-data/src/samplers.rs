//! # Dataset Samplers & Distributed Sharding
//!
//! Sequential, Random, Weighted, and Distributed rank-based shard samplers.

/// Abstract index sampler interface.
pub trait Sampler: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn sample_indices(&self) -> Vec<usize>;
}

/// Sequential in-order index sampler.
pub struct SequentialSampler {
    len: usize,
}

impl SequentialSampler {
    /// Creates a new `SequentialSampler`.
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl Sampler for SequentialSampler {
    fn len(&self) -> usize {
        self.len
    }

    fn sample_indices(&self) -> Vec<usize> {
        (0..self.len).collect()
    }
}

/// Distributed shard-aware sampler.
pub struct DistributedSampler {
    len: usize,
    num_replicas: usize,
    rank: usize,
}

impl DistributedSampler {
    /// Creates a new `DistributedSampler`.
    pub fn new(len: usize, num_replicas: usize, rank: usize) -> Self {
        Self {
            len,
            num_replicas: num_replicas.max(1),
            rank: rank % num_replicas.max(1),
        }
    }
}

impl Sampler for DistributedSampler {
    fn len(&self) -> usize {
        self.len.div_ceil(self.num_replicas)
    }

    fn sample_indices(&self) -> Vec<usize> {
        (self.rank..self.len).step_by(self.num_replicas).collect()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
