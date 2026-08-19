//! # Tensor & Batch Transformation Operations
//!
//! Provides tensor batch manipulation, normalizations, and asynchronous batch mapping helpers.

use crate::core::{Sample, SampleBatch};
use brain_core::Tensor;

/// Transforms a single tensor within a sample.
pub fn transform_sample_tensor<F>(sample: Sample, f: F) -> Sample
where
    F: FnOnce(Tensor) -> Tensor,
{
    Sample {
        id: sample.id,
        data: f(sample.data),
        label: sample.label,
        metadata: sample.metadata,
    }
}

/// Applies a transform function to all samples in a batch.
pub fn transform_batch<F>(batch: SampleBatch, f: F) -> SampleBatch
where
    F: Fn(Sample) -> Sample,
{
    SampleBatch::new(batch.samples.into_iter().map(f).collect())
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
