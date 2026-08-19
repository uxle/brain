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
}
