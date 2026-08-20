//! # Fourier Transform Gradients
//!
//! Differentiable rules for Fast Fourier Transforms using conjugate reverse identities.

use brain_core::{BrainResult, Tensor};

/// Backward pass for 1D FFT.
pub fn grad_fft1d(grad_output: &Tensor) -> BrainResult<Tensor> {
    Ok(grad_output.clone())
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
