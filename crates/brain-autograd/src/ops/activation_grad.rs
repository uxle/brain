//! # Advanced Activation Function Gradients
//!
//! Numerically stable backward implementations for GELU, LeakyReLU, and SiLU.

use brain_core::{BrainResult, Tensor};

/// Backward pass for GELU.
pub fn grad_gelu(x: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let mut d = vec![0.0; x.numel()];
    let x_slice = x.data();
    let g_slice = g.data();
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();

    for (i, (&xi, &gi)) in x_slice.iter().zip(g_slice.iter()).enumerate() {
        let cube = 0.044715 * xi * xi * xi;
        let inner = sqrt_2_over_pi * (xi + cube);
        let tanh_inner = inner.tanh();
        let sech_sq = 1.0 - tanh_inner * tanh_inner;
        let cdf = 0.5 * (1.0 + tanh_inner);
        let pdf = 0.5 * xi * sech_sq * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * xi * xi);
        d[i] = gi * (cdf + pdf);
    }

    Ok(Tensor::from_slice(&d, x.shape().to_vec()))
}

/// Backward pass for LeakyReLU.
pub fn grad_leaky_relu(x: &Tensor, g: &Tensor, negative_slope: f64) -> BrainResult<Tensor> {
    let mut d = vec![0.0; x.numel()];
    for (i, (&xi, &gi)) in x.data().iter().zip(g.data().iter()).enumerate() {
        d[i] = if xi > 0.0 { gi } else { gi * negative_slope };
    }
    Ok(Tensor::from_slice(&d, x.shape().to_vec()))
}

/// Backward pass for SiLU / Swish.
pub fn grad_silu(x: &Tensor, g: &Tensor) -> BrainResult<Tensor> {
    let mut d = vec![0.0; x.numel()];
    for (i, (&xi, &gi)) in x.data().iter().zip(g.data().iter()).enumerate() {
        let sig = 1.0 / (1.0 + (-xi).exp());
        d[i] = gi * (sig + xi * sig * (1.0 - sig));
    }
    Ok(Tensor::from_slice(&d, x.shape().to_vec()))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::tape::OpRecord;
    #[allow(unused_imports)]
    use crate::value::Value;
    #[allow(unused_imports)]
    use brain_core::Tensor;
}
