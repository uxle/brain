//! # Federated Learning Helper Utilities
//!
//! Client sampling, weighted averaging, and round statistics helpers.
#![allow(missing_docs)]

/// Samples a subset of client indices given a fraction.
pub fn sample_clients(num_clients: usize, fraction: f64, seed: u64) -> Vec<usize> {
    let n = ((num_clients as f64) * fraction.clamp(0.0, 1.0)).ceil() as usize;
    let mut rng = seed;
    let mut indices: Vec<usize> = (0..num_clients).collect();
    for i in 0..n {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = i + (rng as usize % (num_clients - i));
        indices.swap(i, j);
    }
    indices[..n].to_vec()
}

/// Computes standard deviation of a slice of f64 values.
pub fn stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    var.sqrt()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
