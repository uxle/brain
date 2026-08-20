//! # Asynchronous Multi-Threaded Prefetching
//!
//! Prefetches batches in background worker threads to overlap compute and IO latencies.

use crate::core::SampleBatch;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

/// Asynchronous prefetch iterator with configurable queue capacity.
pub struct PrefetchIter {
    receiver: Receiver<SampleBatch>,
}

impl PrefetchIter {
    /// Creates a new `PrefetchIter` from a vector of batches with default channel.
    pub fn from_batches(batches: Vec<SampleBatch>) -> Self {
        Self::from_batches_bounded(batches, 4)
    }

    /// Creates a bounded asynchronous prefetch worker with specified buffer capacity.
    pub fn from_batches_bounded(batches: Vec<SampleBatch>, capacity: usize) -> Self {
        let (sender, receiver): (SyncSender<SampleBatch>, Receiver<SampleBatch>) =
            sync_channel(capacity.max(1));

        std::thread::spawn(move || {
            for b in batches {
                if sender.send(b).is_err() {
                    break;
                }
            }
        });
        Self { receiver }
    }

    /// Creates a prefetching stream from any generic batch iterator.
    pub fn from_iterator<I>(iter: I, capacity: usize) -> Self
    where
        I: Iterator<Item = SampleBatch> + Send + 'static,
    {
        let (sender, receiver): (SyncSender<SampleBatch>, Receiver<SampleBatch>) =
            sync_channel(capacity.max(1));

        std::thread::spawn(move || {
            for b in iter {
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
    use super::*;
    use crate::core::Sample;
    use brain_core::Tensor;

    #[test]
    fn test_prefetch_bounded_stream() {
        let b1 = SampleBatch::new(vec![Sample::new(0, Tensor::scalar(1.0))]);
        let b2 = SampleBatch::new(vec![Sample::new(1, Tensor::scalar(2.0))]);

        let mut prefetch = PrefetchIter::from_batches_bounded(vec![b1, b2], 2);
        let first = prefetch.next();
        assert!(first.is_some());
        assert_eq!(first.unwrap().samples[0].id, 0);

        let second = prefetch.next();
        assert!(second.is_some());
        assert_eq!(second.unwrap().samples[0].id, 1);

        assert!(prefetch.next().is_none());
    }
}
