//! # Federated Weight Transforms
//!
//! Normalization, model averaging, Polyak averaging and other weight transforms.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Applies Polyak (exponential moving average) to model weights.
pub fn polyak_average(ema: &[Tensor], new_weights: &[Tensor], momentum: f64) -> Vec<Tensor> {
    let m = Tensor::scalar(momentum);
    let one_minus_m = Tensor::scalar(1.0 - momentum);
    ema.iter()
        .zip(new_weights.iter())
        .map(|(e, n)| &(e * &m) + &(n * &one_minus_m))
        .collect()
}

/// Normalizes each weight tensor to zero mean and unit variance.
pub fn normalize_weights(weights: Vec<Tensor>) -> Vec<Tensor> {
    weights
        .into_iter()
        .map(|t| {
            let data = t.to_vec();
            let n = data.len() as f64;
            if n < 1.0 {
                return t;
            }
            let mean = data.iter().sum::<f64>() / n;
            let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let std = var.sqrt().max(1e-8);
            let norm: Vec<f64> = data.iter().map(|v| (v - mean) / std).collect();
            Tensor::from_vec(norm, t.shape().to_vec())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
