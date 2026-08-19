//! # Federated Aggregation Algorithms
//!
//! FedAvg, FedProx, and adaptive variants for server-side weight aggregation.
#![allow(missing_docs)]

use brain_core::Tensor;
use crate::core::ModelDelta;

/// Supported aggregation algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AggregationAlgorithm {
    #[default]
    FedAvg,
    FedProx,
    FedAdam,
}

/// Aggregates model deltas using FedAvg (sample-weighted average).
pub fn fed_avg_aggregate(deltas: &[ModelDelta]) -> Vec<Tensor> {
    if deltas.is_empty() { return vec![]; }
    let total_samples: usize = deltas.iter().map(|d| d.num_samples).sum();
    if total_samples == 0 { return deltas[0].weights.clone(); }
    let n_layers = deltas[0].weights.len();
    (0..n_layers).map(|l| {
        let mut acc = Tensor::zeros(deltas[0].weights[l].shape().to_vec());
        for d in deltas {
            let w = Tensor::scalar(d.num_samples as f64 / total_samples as f64);
            acc = &acc + &(&d.weights[l] * &w);
        }
        acc
    }).collect()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
