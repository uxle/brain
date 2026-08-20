//! # Pipeline State Checkpointing & Recovery
//!
//! Serializes and restores current epoch and sample offsets for fault-tolerant resumption.

/// Checkpoint state capturing dataset iterator progress.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PipelineCheckpoint {
    pub epoch: usize,
    pub sample_offset: usize,
}

impl PipelineCheckpoint {
    /// Creates a new `PipelineCheckpoint`.
    pub fn new(epoch: usize, sample_offset: usize) -> Self {
        Self {
            epoch,
            sample_offset,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
