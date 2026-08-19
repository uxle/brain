//! # Pipeline Throughput & Latency Metrics
//!
//! Tracks processed samples per second, stage latencies, and buffer queue utilization.

use std::time::Duration;

/// Pipeline execution metrics snapshot.
#[derive(Debug, Clone, Default)]
pub struct PipelineMetrics {
    pub items_processed: usize,
    pub elapsed: Duration,
}

impl PipelineMetrics {
    /// Computes items processed per second.
    pub fn throughput(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs > 0.0 {
            self.items_processed as f64 / secs
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
