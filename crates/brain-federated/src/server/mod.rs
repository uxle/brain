//! # Federated Learning Server
//!
//! Round orchestration, client sampling, and aggregation coordination.
#![allow(missing_docs)]

pub mod aggregate;
pub mod round;

pub use aggregate::{
    fed_avg_aggregate, median_aggregate, trimmed_mean_aggregate, AggregationAlgorithm,
};
pub use round::RoundStats;

use crate::core::RoundId;

/// Configuration for the federated server.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub min_clients: usize,
    pub fraction_fit: f64,
    pub max_rounds: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            min_clients: 2,
            fraction_fit: 1.0,
            max_rounds: 10,
        }
    }
}

/// Federated learning server orchestrating multi-round training.
pub struct FederatedServer {
    pub config: ServerConfig,
    pub current_round: RoundId,
}

impl FederatedServer {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            current_round: 0,
        }
    }

    pub fn advance_round(&mut self) {
        self.current_round += 1;
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
