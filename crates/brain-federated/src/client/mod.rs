//! # Federated Learning Client
//!
//! Local training loop, client configuration, and client reports.
#![allow(missing_docs)]

pub mod trainer;
pub use trainer::LocalTrainer;

use crate::core::{ClientId, ModelDelta};
use brain_core::Tensor;

/// Configuration for a federated client's local training.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub client_id: ClientId,
    pub local_epochs: usize,
    pub learning_rate: f64,
    pub batch_size: usize,
}

impl ClientConfig {
    pub fn new(client_id: ClientId) -> Self {
        Self {
            client_id,
            local_epochs: 5,
            learning_rate: 0.01,
            batch_size: 32,
        }
    }
}

/// Report produced by a client after completing local training.
#[derive(Debug, Clone)]
pub struct ClientReport {
    pub client_id: ClientId,
    pub delta: ModelDelta,
    pub loss: f64,
}

impl ClientReport {
    pub fn new(client_id: ClientId, weights: Vec<Tensor>, num_samples: usize, loss: f64) -> Self {
        Self {
            client_id,
            delta: ModelDelta::new(client_id, weights, num_samples),
            loss,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
