//! # Diffusion Helper Utilities
//!
//! Timestep extraction, linspace grids, and schedule helpers.

use brain_core::Tensor;

/// Extracts a schedule scalar value at timestep `t` and expands to tensor dimensions.
pub fn extract_at_t(schedule_values: &[f64], t: usize) -> Tensor {
    let val = if t < schedule_values.len() {
        schedule_values[t]
    } else {
        0.0
    };
    Tensor::scalar(val)
}

/// Generates a linearly spaced sequence of discrete timesteps.
pub fn linspace_timesteps(total_steps: usize, num_samples: usize) -> Vec<usize> {
    if num_samples <= 1 {
        return vec![0];
    }
    let step = (total_steps - 1) as f64 / (num_samples - 1) as f64;
    (0..num_samples)
        .map(|i| (i as f64 * step).round() as usize)
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
