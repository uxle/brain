//! # Audio Datasets & Waveform Loaders
//!
//! Audio folder scans, paired speech-transcription datasets, and waveform tensor loaders.

use super::Dataset;
use crate::core::Item;
use brain_core::Tensor;

/// In-memory synthetic audio waveform dataset.
pub struct SyntheticAudioDataset {
    num_samples: usize,
    sample_rate: usize,
    duration_secs: f64,
}

impl SyntheticAudioDataset {
    /// Creates a new `SyntheticAudioDataset`.
    pub fn new(num_samples: usize, sample_rate: usize, duration_secs: f64) -> Self {
        Self {
            num_samples,
            sample_rate,
            duration_secs,
        }
    }
}

impl Dataset for SyntheticAudioDataset {
    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, idx: usize) -> Option<Item> {
        if idx < self.num_samples {
            let num_points = (self.sample_rate as f64 * self.duration_secs) as usize;
            let data = Tensor::zeros(vec![1, num_points]);
            Some(Item::new(idx, data).with_target(Tensor::scalar(0.0)))
        } else {
            None
        }
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
