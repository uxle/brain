//! # Multi-Threaded Pipeline Runner
//!
//! Executes epochs and streaming pipelines across thread worker pools.

use crate::core::{DataSource, SampleBatch};

/// Pipeline execution coordinator and runner.
pub struct PipelineRunner;

impl PipelineRunner {
    /// Runs a complete epoch over a data source.
    pub fn run_epoch<D: DataSource>(source: &D, batch_size: usize) -> Vec<SampleBatch> {
        let mut batches = Vec::new();
        let total = source.len();
        let mut cur = Vec::new();

        for i in 0..total {
            if let Some(s) = source.get(i) {
                cur.push(s);
                if cur.len() == batch_size {
                    batches.push(SampleBatch::new(std::mem::take(&mut cur)));
                }
            }
        }
        if !cur.is_empty() {
            batches.push(SampleBatch::new(cur));
        }

        batches
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use crate::core::{DataSource, Sample, SampleBatch};
    use brain_core::Tensor;
}
