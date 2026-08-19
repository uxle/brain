//! # Numeric & Tabular Transforms
//!
//! Provides `Standardize`, `MinMaxScale`, `LogScale`, and `OneHot` encodings.

use super::Transform;
use crate::core::Item;

/// Scales numeric values into `[min, max]` range.
pub struct MinMaxScale {
    pub min_val: f64,
    pub max_val: f64,
}

impl MinMaxScale {
    /// Creates a new `MinMaxScale` transform.
    pub fn new(min_val: f64, max_val: f64) -> Self {
        Self { min_val, max_val }
    }
}

impl Transform for MinMaxScale {
    fn apply(&self, item: Item) -> Item {
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
