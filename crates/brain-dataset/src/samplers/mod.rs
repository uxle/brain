//! # Dataset Samplers
//!
//! Provides `SequentialSampler`, `RandomSampler`, and `BatchSampler`.

/// Abstract dataset sampler trait.
pub trait Sampler: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn sample_indices(&self) -> Vec<usize>;
}

/// Sequential index sampler.
pub struct SequentialSampler {
    pub len: usize,
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

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::Item;
    use crate::dataset::Dataset;
    use brain_core::Tensor;
}
