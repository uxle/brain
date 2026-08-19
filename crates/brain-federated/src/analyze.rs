//! # Federated Analysis Utilities
//!
//! Convergence analysis, data heterogeneity metrics, and communication cost estimation.
#![allow(missing_docs)]

use crate::core::ModelDelta;

/// Computes cosine similarity between two flat delta vectors.
pub fn cosine_similarity_deltas(a: &ModelDelta, b: &ModelDelta) -> f64 {
    if a.weights.is_empty() || b.weights.is_empty() { return 0.0; }
    let va: Vec<f64> = a.weights.iter().flat_map(|t| t.to_vec()).collect();
    let vb: Vec<f64> = b.weights.iter().flat_map(|t| t.to_vec()).collect();
    let dot: f64 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
    let na: f64 = va.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nb: f64 = vb.iter().map(|v| v * v).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 { 0.0 } else { dot / (na * nb) }
}

/// Estimates data heterogeneity (earth mover's distance proxy) across clients.
pub fn estimate_heterogeneity(deltas: &[ModelDelta]) -> f64 {
    if deltas.len() < 2 { return 0.0; }
    let norms: Vec<f64> = deltas.iter().map(|d| {
        d.weights.iter().flat_map(|t| t.to_vec()).map(|v| v * v).sum::<f64>().sqrt()
    }).collect();
    let mean = norms.iter().sum::<f64>() / norms.len() as f64;
    norms.iter().map(|n| (n - mean).abs()).sum::<f64>() / norms.len() as f64
}

/// Communication cost in approximate bytes for delta transmission.
pub fn communication_cost_bytes(delta: &ModelDelta) -> usize {
    let floats: usize = delta.weights.iter().map(|t| t.to_vec().len()).sum();
    floats * 4 // f32 bytes
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
