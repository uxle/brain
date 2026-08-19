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
}
