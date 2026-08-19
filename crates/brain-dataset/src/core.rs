//! # Core Dataset Types & Items
//!
//! Provides fundamental [`Item`] and [`Batch`] abstractions.

use brain_core::Tensor;

/// A single dataset sample item containing payload tensor and optional label.
#[derive(Debug, Clone)]
pub struct Item {
    pub id: usize,
    pub data: Tensor,
    pub target: Option<Tensor>,
}

impl Item {
    /// Creates a new `Item`.
    pub fn new(id: usize, data: Tensor) -> Self {
        Self {
            id,
            data,
            target: None,
        }
    }

    /// Attaches a target label to the item.
    pub fn with_target(mut self, target: Tensor) -> Self {
        self.target = Some(target);
        self
    }
}

/// A contiguous batch of dataset items.
#[derive(Debug, Clone)]
pub struct Batch {
    pub items: Vec<Item>,
}

impl Batch {
    /// Creates a new `Batch`.
    pub fn new(items: Vec<Item>) -> Self {
        Self { items }
    }

    /// Returns the number of items in the batch.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
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
