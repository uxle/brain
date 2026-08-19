//! # Multi-Source Pipeline Combinators
//!
//! Concatenates, zips, and interleaves multiple distinct data streams.

use crate::core::{DataSource, Sample};

/// Concatenates multiple data sources sequentially.
pub struct ConcatSources<A, B> {
    source_a: A,
    source_b: B,
}

impl<A: DataSource, B: DataSource> ConcatSources<A, B> {
    /// Creates a new `ConcatSources`.
    pub fn new(source_a: A, source_b: B) -> Self {
        Self { source_a, source_b }
    }
}

impl<A: DataSource, B: DataSource> DataSource for ConcatSources<A, B> {
    fn len(&self) -> usize {
        self.source_a.len() + self.source_b.len()
    }

    fn get(&self, idx: usize) -> Option<Sample> {
        let len_a = self.source_a.len();
        if idx < len_a {
            self.source_a.get(idx)
        } else {
            self.source_b.get(idx - len_a)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
