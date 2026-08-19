//! # Dataset Batch Operations
//!
//! Functional batch transformations and filter predicates.

use crate::core::{Batch, Item};

/// Applies an in-place transformation across all items in a batch.
pub fn map_batch<F>(batch: Batch, f: F) -> Batch
where
    F: Fn(Item) -> Item,
{
    Batch::new(batch.items.into_iter().map(f).collect())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::Item;
    use crate::dataset::Dataset;
    use brain_core::Tensor;
}
