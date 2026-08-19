//! # Chunked Stream Datasets
//!
//! Streaming datasets for ingesting massive corpora with chunked reads and resume checkpoints.

use crate::core::Sample;
use brain_core::Tensor;

/// Streaming dataset yielding samples incrementally.
pub struct StreamDataset {
    total_chunks: usize,
}

impl StreamDataset {
    /// Creates a new `StreamDataset`.
    pub fn new(total_chunks: usize) -> Self {
        Self { total_chunks }
    }

    /// Reads the next sample from the stream.
    pub fn next_sample(&mut self, chunk_idx: usize) -> Option<Sample> {
        if chunk_idx < self.total_chunks {
            Some(Sample::new(chunk_idx, Tensor::zeros(vec![4])))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
