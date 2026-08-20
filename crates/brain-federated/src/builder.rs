//! # Federated System Builder
//!
//! Ergonomic builder for assembling a complete federated learning system.
#![allow(missing_docs)]

use crate::client::ClientConfig;
use crate::server::{FederatedServer, ServerConfig};

/// Builder for constructing a federated learning system.
#[derive(Debug, Default)]
pub struct FedSystemBuilder {
    num_clients: usize,
    rounds: usize,
    fraction_fit: f64,
    local_epochs: usize,
}

impl FedSystemBuilder {
    pub fn new() -> Self {
        Self {
            num_clients: 10,
            rounds: 10,
            fraction_fit: 1.0,
            local_epochs: 5,
        }
    }

    pub fn num_clients(mut self, n: usize) -> Self {
        self.num_clients = n;
        self
    }
    pub fn rounds(mut self, r: usize) -> Self {
        self.rounds = r;
        self
    }
    pub fn fraction_fit(mut self, f: f64) -> Self {
        self.fraction_fit = f;
        self
    }
    pub fn local_epochs(mut self, e: usize) -> Self {
        self.local_epochs = e;
        self
    }

    pub fn build_server(self) -> FederatedServer {
        FederatedServer::new(ServerConfig {
            min_clients: 2,
            fraction_fit: self.fraction_fit,
            max_rounds: self.rounds,
        })
    }

    pub fn build_client_configs(&self) -> Vec<ClientConfig> {
        (0..self.num_clients)
            .map(|id| ClientConfig {
                client_id: id,
                local_epochs: self.local_epochs,
                learning_rate: 0.01,
                batch_size: 32,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
