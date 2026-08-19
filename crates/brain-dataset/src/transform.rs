//! # Composed Transform Graph
//!
//! Graph execution engine for transforming dataset samples.

use crate::core::Item;

/// Transform graph runner.
#[derive(Default)]
pub struct TransformGraph {
    stages: usize,
}

impl TransformGraph {
    /// Creates a new `TransformGraph`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a stage.
    pub fn add_stage(mut self) -> Self {
        self.stages += 1;
        self
    }

    /// Executes transformation on an item.
    pub fn execute(&self, item: Item) -> Item {
        item
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
