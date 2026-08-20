//! # Abstract Dataset Interface & Combinators
//!
//! Provides the primary [`Dataset`] trait, [`MapDataset`], [`ConcatDataset`], and [`Subset`].

pub mod audio;
pub mod tabular;
pub mod text;
pub mod vision;
pub mod vision_v2;

use crate::core::Item;

/// Abstract random-access dataset.
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, idx: usize) -> Option<Item>;
}

/// Slices a subset of an underlying dataset.
pub struct Subset<'a, D: Dataset> {
    pub dataset: &'a D,
    pub indices: Vec<usize>,
}

impl<'a, D: Dataset> Subset<'a, D> {
    /// Creates a new `Subset`.
    pub fn new(dataset: &'a D, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }
}

impl<'a, D: Dataset> Dataset for Subset<'a, D> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, idx: usize) -> Option<Item> {
        self.indices
            .get(idx)
            .and_then(|&real_idx| self.dataset.get(real_idx))
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
