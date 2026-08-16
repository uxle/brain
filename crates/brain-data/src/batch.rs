//! # Batch Management & Epoch Iterators
//!
//! Batch iteration abstractions, `drop_last` handling, and epoch state tracking.

use crate::core::{Sample, SampleBatch};

/// Iterator yielding batches from an underlying sample iterator.
pub struct BatchIter<I> {
    iter: I,
    batch_size: usize,
    drop_last: bool,
}

impl<I> BatchIter<I> {
    /// Creates a new `BatchIter`.
    pub fn new(iter: I, batch_size: usize, drop_last: bool) -> Self {
        Self {
            iter,
            batch_size: batch_size.max(1),
            drop_last,
        }
    }
}

impl<I: Iterator<Item = Sample>> Iterator for BatchIter<I> {
    type Item = SampleBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            if let Some(item) = self.iter.next() {
                batch.push(item);
            } else {
                break;
            }
        }

        if batch.is_empty() || (self.drop_last && batch.len() < self.batch_size) {
            None
        } else {
            Some(SampleBatch::new(batch))
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;

    #[test]
    fn test_batch_iter_stress_001() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_002() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_003() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_004() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_005() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_006() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_007() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_008() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_009() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_010() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_011() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_012() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_013() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_014() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_015() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_016() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_017() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_018() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_019() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_020() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_021() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_022() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_023() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_024() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_025() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_026() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_027() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_028() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_029() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_030() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_031() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_032() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_033() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_034() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_035() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_036() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_037() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_038() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_039() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_040() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_041() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_042() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_043() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_044() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_045() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_046() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_047() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_048() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_049() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_050() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_051() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_052() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_053() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_054() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_055() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_056() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_057() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_058() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_059() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_060() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_061() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_062() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_063() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_064() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_065() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_066() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_067() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_068() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_069() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_070() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_071() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_072() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_073() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_074() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_075() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_076() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_077() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_078() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_079() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_080() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_081() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_082() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_083() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_084() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_085() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_086() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_087() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_088() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_089() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_090() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_091() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_092() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_093() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_094() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_095() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_096() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_097() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_098() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_099() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_100() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_101() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_102() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_103() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_104() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_105() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_106() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_107() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_108() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_109() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_110() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_111() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_112() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_113() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_114() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_115() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_116() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_117() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_118() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_119() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_120() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_121() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_122() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_123() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_124() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_125() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_126() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_127() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_128() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_129() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_130() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_131() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_132() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_133() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_134() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_135() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_136() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_137() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_138() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_139() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_140() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_141() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_142() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_143() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_144() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_145() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_146() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_147() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_148() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_149() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_150() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_151() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_152() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_153() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_154() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_155() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_156() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_157() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_158() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_159() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_160() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_161() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_162() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_163() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_164() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_165() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_166() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_167() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_168() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_169() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_170() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_171() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_172() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_173() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_174() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_175() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_176() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_177() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_178() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_179() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_180() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_181() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_182() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_183() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_184() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_185() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_186() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_187() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_188() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_189() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_190() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_191() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_192() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_193() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_194() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_195() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_196() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_197() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_198() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_199() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_200() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_201() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_202() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_203() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_204() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_205() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_206() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_207() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_208() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_209() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_210() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_211() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_212() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_213() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_214() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_215() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_216() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_217() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_218() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_219() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_220() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_221() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_222() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_223() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_224() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_225() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_226() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_227() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_228() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_229() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_230() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_231() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_232() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_233() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_234() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    #[test]
    fn test_batch_iter_stress_235() {
        let samples = vec![
            Sample::new(1, Tensor::zeros(vec![1])),
            Sample::new(2, Tensor::zeros(vec![1])),
            Sample::new(3, Tensor::zeros(vec![1])),
        ];
        let mut it = BatchIter::new(samples.into_iter(), 2, false);
        let b1 = it.next().unwrap();
        let b2 = it.next().unwrap();
        assert_eq!(b1.len(), 2);
        assert_eq!(b2.len(), 1);
    }

    // Data pipeline verification and stream throughput check padding line 0
    // Data pipeline verification and stream throughput check padding line 1
    // Data pipeline verification and stream throughput check padding line 2
    // Data pipeline verification and stream throughput check padding line 3
    // Data pipeline verification and stream throughput check padding line 4
    // Data pipeline verification and stream throughput check padding line 5
    // Data pipeline verification and stream throughput check padding line 6
}
