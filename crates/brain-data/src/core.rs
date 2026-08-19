//! # Core Data Pipeline Types & Source Abstractions
//!
//! Provides the primary [`Sample`], [`SampleBatch`], [`DataSource`], and [`DataReader`] abstractions.

use brain_core::Tensor;
use std::collections::HashMap;

/// A single data sample with associated tensor payload and metadata.
#[derive(Debug, Clone)]
pub struct Sample {
    pub id: usize,
    pub data: Tensor,
    pub label: Option<Tensor>,
    pub metadata: HashMap<String, String>,
}

impl Sample {
    /// Creates a new `Sample`.
    pub fn new(id: usize, data: Tensor) -> Self {
        Self {
            id,
            data,
            label: None,
            metadata: HashMap::new(),
        }
    }

    /// Attaches a label tensor to the sample.
    pub fn with_label(mut self, label: Tensor) -> Self {
        self.label = Some(label);
        self
    }

    /// Attaches metadata key-value pair to the sample.
    pub fn with_meta(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), val.into());
        self
    }
}

/// A contiguous batch of aggregated samples.
#[derive(Debug, Clone)]
pub struct SampleBatch {
    pub samples: Vec<Sample>,
}

impl SampleBatch {
    /// Creates a new `SampleBatch`.
    pub fn new(samples: Vec<Sample>) -> Self {
        Self { samples }
    }

    /// Returns the number of samples in the batch.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

/// Abstract random-access data source.
pub trait DataSource: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, idx: usize) -> Option<Sample>;
}

/// Abstract contiguous data reader.
pub trait DataReader: Send + Sync {
    fn read_batch(&self, indices: &[usize]) -> Vec<Sample>;
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
