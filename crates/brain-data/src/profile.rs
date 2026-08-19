//! # Pipeline Latency Profiling
//!
//! Measures per-stage compute and wait times to identify throughput bottlenecks.

use std::time::Duration;

/// Diagnostic profile report for a pipeline stage.
#[derive(Debug, Clone, Default)]
pub struct StageProfile {
    pub name: String,
    pub execution_time: Duration,
}

impl StageProfile {
    /// Creates a new `StageProfile`.
    pub fn new(name: impl Into<String>, execution_time: Duration) -> Self {
        Self {
            name: name.into(),
            execution_time,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
