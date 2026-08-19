//! # Pipeline Stage Library
//!
//! Provides `MapStage`, `FilterStage`, `BatchStage`, `ShuffleStage`, and `PrefetchStage`.

use crate::core::Sample;

/// Pipeline processing stage trait.
pub trait Stage: Send + Sync {
    fn name(&self) -> &str;
    fn process(&self, sample: Sample) -> Option<Sample>;
}

/// Map transformation stage applying a mapping closure to each sample.
pub struct MapStage<F> {
    pub name: String,
    pub func: F,
}

impl<F> MapStage<F>
where
    F: Fn(Sample) -> Sample + Send + Sync,
{
    /// Creates a new `MapStage`.
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
        }
    }
}

impl<F> Stage for MapStage<F>
where
    F: Fn(Sample) -> Sample + Send + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&self, sample: Sample) -> Option<Sample> {
        Some((self.func)(sample))
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
