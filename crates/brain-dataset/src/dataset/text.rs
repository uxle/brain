//! # Text Dataset Parsers & Corpus Iterators
//!
//! Provides `TextFileDataset`, `CsvDataset`, and tokenized text sequence loaders.

use super::Dataset;
use crate::core::Item;
use brain_core::Tensor;

/// In-memory text lines dataset.
pub struct TextLinesDataset {
    lines: Vec<String>,
}

impl TextLinesDataset {
    /// Creates a new `TextLinesDataset` from a vector of strings.
    pub fn new(lines: Vec<String>) -> Self {
        Self { lines }
    }
}

impl Dataset for TextLinesDataset {
    fn len(&self) -> usize {
        self.lines.len()
    }

    fn get(&self, idx: usize) -> Option<Item> {
        self.lines.get(idx).map(|line| {
            let tokens = line.as_bytes().iter().map(|&b| b as f64).collect();
            Item::new(idx, Tensor::from_vec(tokens, vec![line.len()]))
        })
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
