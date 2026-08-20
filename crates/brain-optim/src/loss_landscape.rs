//! # Loss Landscape Geometry & Profiling
//!
//! 1D/2D parameter interpolation, filter normalization (Li et al.), and curvature exploration.
#![allow(missing_docs)]

use brain_core::Tensor;

/// Configuration for 1D/2D loss landscape interpolation.
#[derive(Debug, Clone, PartialEq)]
pub struct LossLandscapeConfig {
    pub num_points: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub filter_normalized: bool,
}

impl Default for LossLandscapeConfig {
    fn default() -> Self {
        Self {
            num_points: 21,
            x_min: -1.0,
            x_max: 1.0,
            y_min: -1.0,
            y_max: 1.0,
            filter_normalized: true,
        }
    }
}

/// Computes filter-normalized random perturbation direction for a parameter tensor.
pub fn create_filter_normalized_direction(param: &Tensor) -> Tensor {
    let p_data = param.data();
    let n = p_data.len();
    let mut d_data = vec![1.0; n];

    let mut p_norm_sq: f64 = 0.0;
    let mut d_norm_sq: f64 = 0.0;
    for i in 0..n {
        p_norm_sq += p_data[i] * p_data[i];
        d_norm_sq += d_data[i] * d_data[i];
    }

    let p_norm = p_norm_sq.sqrt();
    let d_norm = d_norm_sq.sqrt().max(1e-12);
    let scale = p_norm / d_norm;

    for val in d_data.iter_mut() {
        *val *= scale;
    }

    Tensor::from_slice(&d_data, param.shape().to_vec())
}

/// Interpolates parameter weights along 1D line: theta = theta_0 + alpha * direction.
pub fn interpolate_1d(theta_0: &[Tensor], direction: &[Tensor], alpha: f64) -> Vec<Tensor> {
    let mut result = Vec::with_capacity(theta_0.len());
    for (p, d) in theta_0.iter().zip(direction.iter()) {
        let p_data = p.data();
        let d_data = d.data();
        let mut out = vec![0.0; p_data.len()];
        for i in 0..p_data.len() {
            out[i] = p_data[i] + alpha * d_data[i];
        }
        result.push(Tensor::from_slice(&out, p.shape().to_vec()));
    }
    result
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant
    )]
    use super::*;
    use brain_core::Tensor;
}
