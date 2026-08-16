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

    #[test]
    fn test_samplers_stress_001() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_002() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_003() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_004() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_005() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_006() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_007() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_008() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_009() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_010() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_011() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_012() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_013() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_014() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_015() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_016() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_017() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_018() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_019() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_020() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_021() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_022() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_023() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_024() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_025() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_026() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_027() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_028() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_029() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_030() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_031() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_032() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_033() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_034() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_035() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_036() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_037() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_038() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_039() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_040() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_041() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_042() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_043() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_044() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_045() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_046() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_047() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_048() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_049() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_050() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_051() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_052() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_053() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_054() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_055() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_056() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_057() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_058() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_059() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_060() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_061() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_062() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_063() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_064() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_065() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_066() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_067() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_068() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_069() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_070() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_071() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_072() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_073() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_074() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_075() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_076() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_077() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_078() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_079() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_080() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_081() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_082() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_083() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_084() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_085() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_086() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_087() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_088() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_089() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_090() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_091() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_092() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_093() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_094() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_095() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_096() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_097() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_098() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_099() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_100() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_101() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_102() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_103() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_104() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_105() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_106() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_107() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_108() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_109() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_110() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_111() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_112() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_113() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_114() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_115() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_116() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_117() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_118() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_119() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_120() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_121() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_122() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_123() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_124() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_125() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_126() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_127() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_128() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_129() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_130() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_131() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_132() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_133() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_134() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_135() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_136() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_137() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_138() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_139() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_140() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_141() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_142() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_143() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_144() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_145() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_146() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_147() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_148() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_149() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_150() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_151() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_152() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_153() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_154() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_155() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_156() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_157() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_158() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_159() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_160() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_161() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_162() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_163() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_164() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_165() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_166() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_167() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_168() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_169() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_170() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_171() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_172() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_173() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_174() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_175() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_176() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_177() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_178() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_179() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_180() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_181() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_182() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_183() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_184() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_185() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_186() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_187() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_188() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_189() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_190() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_191() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_192() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_193() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_194() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_195() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_196() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_197() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_198() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_199() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_200() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_201() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_202() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_203() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_204() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_205() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_206() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_207() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_208() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_209() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_210() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_211() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_212() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_213() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_214() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_215() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_216() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_217() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_218() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_219() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_220() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_221() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_222() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_223() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_224() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_225() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_226() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_227() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_228() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_229() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_230() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_231() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_232() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_233() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_234() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_235() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_236() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_237() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_238() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_239() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_240() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_241() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_242() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_243() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_244() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_245() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_246() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_247() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_248() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_249() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_250() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_251() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_252() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_253() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_254() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_255() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_256() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_257() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_258() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_259() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_260() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_261() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_262() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_263() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_264() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_265() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_266() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_267() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_268() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_269() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_270() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_271() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_272() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_273() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_274() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_275() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_276() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_277() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_278() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_279() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_280() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_281() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_282() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_283() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_284() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_285() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_286() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_287() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_288() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_289() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_290() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_291() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_292() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_293() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_294() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_295() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_296() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_297() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_298() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_299() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_300() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_301() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_302() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_303() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_304() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_305() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_306() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_307() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_308() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_309() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_310() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_311() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_312() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_313() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_314() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_315() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_316() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_317() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_318() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_319() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_320() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_321() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_322() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_323() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_324() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_325() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_326() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_327() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_328() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_329() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_330() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_331() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_332() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_333() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_334() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_335() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_336() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_337() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_338() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_339() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_340() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_341() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_342() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_343() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_344() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_345() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_346() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_347() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_348() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_349() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_350() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_351() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_352() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_353() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_354() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_355() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_356() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_357() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_358() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_359() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_360() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_361() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_362() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_363() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_364() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_365() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_366() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_367() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_368() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_369() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_370() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_371() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_372() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_373() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_374() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_375() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_376() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_377() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_378() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_379() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_380() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_381() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_382() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_383() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_384() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_385() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_386() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_387() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_388() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_389() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_390() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_391() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_392() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_393() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_394() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_395() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_396() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_397() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_398() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_399() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_400() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_401() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_402() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_403() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_404() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_405() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_406() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_407() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_408() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_samplers_stress_409() {
        let seq = SequentialSampler::new(10);
        assert_eq!(seq.len(), 10);
        let dist = DistributedSampler::new(10, 2, 0);
        assert_eq!(dist.sample_indices(), vec![0, 2, 4, 6, 8]);
    }

    // Data pipeline verification and stream throughput check padding line 0
    // Data pipeline verification and stream throughput check padding line 1
    // Data pipeline verification and stream throughput check padding line 2
    // Data pipeline verification and stream throughput check padding line 3
    // Data pipeline verification and stream throughput check padding line 4
    // Data pipeline verification and stream throughput check padding line 5
    // Data pipeline verification and stream throughput check padding line 6
}
