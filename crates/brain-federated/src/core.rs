//! # Core Federated Learning Types
//!
//! Provides [`ClientId`], [`RoundId`], [`ModelDelta`], and associated metrics.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Unique identifier for a federated client.
pub type ClientId = usize;
/// Unique identifier for a training round.
pub type RoundId = usize;

/// Weight update delta from a single client's local training.
#[derive(Debug, Clone)]
pub struct ModelDelta {
    pub client_id: ClientId,
    pub weights: Vec<Tensor>,
    pub num_samples: usize,
}

impl ModelDelta {
    pub fn new(client_id: ClientId, weights: Vec<Tensor>, num_samples: usize) -> Self {
        Self {
            client_id,
            weights,
            num_samples,
        }
    }
}

/// Per-client training metrics reported after a local round.
#[derive(Debug, Clone, Default)]
pub struct ClientMetrics {
    pub loss: f64,
    pub accuracy: f64,
    pub num_samples: usize,
}

/// Server-side global metrics after aggregation.
#[derive(Debug, Clone, Default)]
pub struct ServerMetrics {
    pub round_id: RoundId,
    pub global_loss: f64,
    pub participating_clients: usize,
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
