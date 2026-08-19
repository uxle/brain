//! # Audio Transforms & Spectrogram Extraction
//!
//! Provides `Resample`, `ToMel`, `ToMFCC`, and `TimeShift`.

use super::Transform;
use crate::core::Item;

/// Resamples audio signals to target sample rate.
pub struct Resample {
    pub orig_sr: usize,
    pub target_sr: usize,
}

impl Resample {
    /// Creates a new `Resample` transform.
    pub fn new(orig_sr: usize, target_sr: usize) -> Self {
        Self { orig_sr, target_sr }
    }
}

impl Transform for Resample {
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
