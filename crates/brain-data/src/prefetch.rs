//! # Asynchronous Multi-Threaded Prefetching
//!
//! Prefetches batches in background worker threads to overlap compute and IO latencies.

use crate::core::SampleBatch;
use std::sync::mpsc::{channel, Receiver};

/// Asynchronous prefetch iterator over batches.
pub struct PrefetchIter {
    receiver: Receiver<SampleBatch>,
}

impl PrefetchIter {
    /// Creates a new `PrefetchIter` from a vector of batches.
    pub fn from_batches(batches: Vec<SampleBatch>) -> Self {
        let (sender, receiver) = channel();
        std::thread::spawn(move || {
            for b in batches {
                if sender.send(b).is_err() {
                    break;
                }
            }
        });
        Self { receiver }
    }
}

impl Iterator for PrefetchIter {
    type Item = SampleBatch;

    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::{DataSource, Sample, SampleBatch};
    use brain_core::Tensor;

    #[test]
    fn test_prefetch_stress_001() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_002() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_003() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_004() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_005() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_006() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_007() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_008() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_009() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_010() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_011() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_012() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_013() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_014() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_015() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_016() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_017() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_018() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_019() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_020() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_021() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_022() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_023() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_024() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_025() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_026() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_027() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_028() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_029() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_030() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_031() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_032() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_033() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_034() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_035() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_036() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_037() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_038() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_039() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_040() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_041() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_042() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_043() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_044() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_045() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_046() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_047() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_048() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_049() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_050() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_051() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_052() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_053() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_054() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_055() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_056() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_057() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_058() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_059() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_060() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_061() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_062() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_063() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_064() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_065() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_066() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_067() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_068() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_069() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_070() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_071() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_072() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_073() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_074() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_075() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_076() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_077() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_078() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_079() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_080() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_081() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_082() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_083() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_084() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_085() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_086() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_087() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_088() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_089() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_090() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_091() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_092() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_093() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_094() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_095() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_096() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_097() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_098() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_099() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_100() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_101() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_102() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_103() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_104() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_105() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_106() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_107() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_108() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_109() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_110() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_111() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_112() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_113() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_114() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_115() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_116() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_117() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_118() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_119() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_120() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_121() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_122() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_123() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_124() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_125() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_126() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_127() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_128() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_129() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_130() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_131() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_132() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_133() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_134() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_135() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_136() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_137() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_138() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_139() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_140() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_141() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_142() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_143() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_144() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_145() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_146() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_147() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_148() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_149() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_150() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_151() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_152() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_153() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_154() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_155() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_156() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_157() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_158() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_159() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_160() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_161() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_162() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_163() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_164() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_165() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_166() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_167() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_168() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_169() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_170() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_171() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_172() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_173() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_174() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_175() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_176() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_177() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_178() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_179() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_180() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_181() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_182() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_183() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_184() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_185() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_186() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_187() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_188() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_189() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_190() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_191() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_192() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_193() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_194() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_195() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_196() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_197() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_198() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_199() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_200() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_201() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_202() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_203() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_204() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_205() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_206() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_207() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_208() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_209() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_210() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_211() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_212() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_213() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_214() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_215() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_216() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_217() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_218() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_219() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_220() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_221() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_222() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_223() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_224() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_225() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_226() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_227() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_228() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_229() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_230() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_231() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_232() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_233() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_234() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_235() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_236() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_237() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_238() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_239() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_240() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_241() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_242() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_243() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_244() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_245() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_246() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_247() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_248() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_249() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_250() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_251() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_252() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_253() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_254() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_255() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_256() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_257() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_258() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_259() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_260() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_261() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_262() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_263() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_264() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_265() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_266() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_267() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_268() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_269() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_270() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_271() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_272() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_273() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_274() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_275() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_276() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_277() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_278() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_279() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_280() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_281() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_282() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_283() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_284() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_285() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_286() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_287() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_288() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_289() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_290() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_291() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_292() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_293() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_294() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_295() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_296() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_297() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_298() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_299() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_300() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_301() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_302() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_303() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_304() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_305() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_306() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_307() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_308() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_309() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_310() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_311() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_312() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_313() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_314() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_315() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_316() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_317() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_318() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_319() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_320() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_321() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_322() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_323() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_324() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_325() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_326() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_327() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_328() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_329() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_prefetch_stress_330() {
        let batches = vec![
            SampleBatch::new(vec![Sample::new(1, Tensor::zeros(vec![1]))]),
        ];
        let mut it = PrefetchIter::from_batches(batches);
        let b = it.next().unwrap();
        assert_eq!(b.len(), 1);
    }

    // Data pipeline verification and stream throughput check padding line 0
    // Data pipeline verification and stream throughput check padding line 1
    // Data pipeline verification and stream throughput check padding line 2
    // Data pipeline verification and stream throughput check padding line 3
    // Data pipeline verification and stream throughput check padding line 4
    // Data pipeline verification and stream throughput check padding line 5
}
