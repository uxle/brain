//! # Data Transformation Pipelines & Traits
//!
//! Provides the primary [`Transform`] trait and composable [`Compose`] pipeline.

pub mod audio_t;
pub mod numeric_t;
pub mod text_t;
pub mod vision_t;

use crate::core::Item;

/// Abstract data transformation trait.
pub trait Transform: Send + Sync {
    fn apply(&self, item: Item) -> Item;
}

/// Composed sequential transformation pipeline.
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Default for Compose {
    fn default() -> Self {
        Self::new()
    }
}

impl Compose {
    /// Creates an empty `Compose` pipeline.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Appends a transform to the pipeline.
    #[allow(clippy::should_implement_trait)]
    pub fn add<T: Transform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }
}

impl Transform for Compose {
    fn apply(&self, mut item: Item) -> Item {
        for t in &self.transforms {
            item = t.apply(item);
        }
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
