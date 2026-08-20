//! # Quantization & Straight-Through Estimator (STE)
//!
//! Differentiable approximations for quantized operations.

use brain_core::{BrainResult, Tensor};

/// Straight-Through Estimator (STE) backward for hard quantization.
pub fn grad_quantize_ste(
    x: &Tensor,
    g: &Tensor,
    min_val: f64,
    max_val: f64,
) -> BrainResult<Tensor> {
    let mut d = vec![0.0; x.numel()];
    for (i, (&xi, &gi)) in x.data().iter().zip(g.data().iter()).enumerate() {
        if xi >= min_val && xi <= max_val {
            d[i] = gi;
        }
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
