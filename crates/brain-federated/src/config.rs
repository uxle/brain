//! # Federated Learning Configuration
//!
//! Master configuration struct covering server, client, transport and privacy settings.
#![allow(missing_docs)]

/// Complete federated learning system configuration.
#[derive(Debug, Clone)]
pub struct FedConfig {
    pub num_clients: usize,
    pub rounds: usize,
    pub fraction_fit: f64,
    pub local_epochs: usize,
    pub learning_rate: f64,
}

impl Default for FedConfig {
    fn default() -> Self {
        Self {
            num_clients: 10,
            rounds: 10,
            fraction_fit: 1.0,
            local_epochs: 5,
            learning_rate: 0.01,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
