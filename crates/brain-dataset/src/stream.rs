//! # Streaming Datasets
//!
//! Incremental parsers for streaming records with bounded memory footprints.

use crate::core::Item;
use brain_core::Tensor;

/// Incremental streaming reader.
pub struct StreamingReader {
    current_idx: usize,
}

impl Default for StreamingReader {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingReader {
    /// Creates a new `StreamingReader`.
    pub fn new() -> Self {
        Self { current_idx: 0 }
    }

    /// Reads next stream item.
    pub fn next_item(&mut self) -> Option<Item> {
        let idx = self.current_idx;
        self.current_idx += 1;
        Some(Item::new(idx, Tensor::zeros(vec![1])))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::Item;
    use crate::dataset::Dataset;
    use brain_core::Tensor;
}
