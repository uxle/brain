//! # Sinusoidal & Learned Timestep Embeddings
//!
//! Converts discrete integer timesteps into continuous sinusoidal frequency vectors.

use brain_core::Tensor;

/// Generates sinusoidal timestep embeddings for a sequence of timesteps.
pub fn sinusoidal_timestep_embedding(timesteps: &[usize], dim: usize) -> Tensor {
    let mut data = Vec::with_capacity(timesteps.len() * dim);
    let half_dim = dim / 2;

    for &t in timesteps {
        for i in 0..half_dim {
            let freq = (-((i as f64) / (half_dim as f64) * (10000.0_f64).ln())).exp();
            let arg = t as f64 * freq;
            data.push(arg.sin());
            data.push(arg.cos());
        }
        if dim % 2 == 1 {
            data.push(0.0);
        }
    }

    Tensor::from_vec(data, vec![timesteps.len(), dim])
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code)]
    use super::*;
    use brain_core::Tensor;
}
