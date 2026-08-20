//! # Image & Bounding Box Augmentation Pipeline
//!
//! Provides image transforms, composable pipelines (`Compose`), and reproducible random seeds.

pub mod boxes;
pub mod color;
pub mod geom;
pub mod mix;
pub mod photo;

pub use boxes::transform_bounding_boxes;
pub use color::ColorJitter;
pub use geom::RandomResizedCrop;
pub use mix::{cutmix, mixup, sample_cutmix_box};
pub use photo::solarize;

use brain_core::Tensor;

/// Composable image transformation pipeline.
#[derive(Default)]
pub struct Compose {
    transforms: Vec<Box<dyn Fn(&Tensor) -> Tensor + Send + Sync>>,
}

impl Compose {
    /// Creates an empty `Compose` pipeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a transform closure to the pipeline.
    pub fn add<F>(&mut self, transform: F)
    where
        F: Fn(&Tensor) -> Tensor + Send + Sync + 'static,
    {
        self.transforms.push(Box::new(transform));
    }

    /// Applies the sequential transforms to an image tensor.
    pub fn apply(&self, input: &Tensor) -> Tensor {
        let mut cur = input.clone();
        for t in &self.transforms {
            cur = t(&cur);
        }
        cur
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
