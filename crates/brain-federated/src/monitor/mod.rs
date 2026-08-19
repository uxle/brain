//! # Federated Training Monitor
//!
//! Round metrics, convergence detection, and training history.
#![allow(missing_docs)]

use crate::server::round::RoundStats;

/// Monitor accumulating per-round statistics.
#[derive(Debug, Default)]
pub struct FedMonitor {
    pub history: Vec<RoundStats>,
}

impl FedMonitor {
    pub fn new() -> Self { Self::default() }

    pub fn record(&mut self, stats: RoundStats) {
        self.history.push(stats);
    }

    pub fn latest_loss(&self) -> Option<f64> {
        self.history.last().map(|s| s.avg_loss)
    }

    pub fn has_converged(&self, patience: usize, tolerance: f64) -> bool {
        if self.history.len() < patience { return false; }
        let n = self.history.len();
        let recent = &self.history[n - patience..];
        let losses: Vec<f64> = recent.iter().map(|s| s.avg_loss).collect();
        let range = losses.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - losses.iter().copied().fold(f64::INFINITY, f64::min);
        range < tolerance
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
