//! # Vectorized Regularization Mathematical Kernels
//!
//! Low-level mathematical routines for dropout scaling, batch statistics, and layer normalizations.
#![allow(
    missing_docs,
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::doc_markdown
)]

/// Inverted dropout scaling kernel.
pub fn dropout_apply(input: &[f64], mask: &[f64], p: f64) -> Vec<f64> {
    let scale = if p < 1.0 { 1.0 / (1.0 - p) } else { 0.0 };
    let mut out = Vec::with_capacity(input.len());
    for i in 0..input.len() {
        out.push(input[i] * mask[i] * scale);
    }
    out
}

/// Vectorized normalization kernel given mean, variance, scale (gamma), and shift (beta).
pub fn norm_apply_affine(
    input: &[f64],
    mean: f64,
    var: f64,
    eps: f64,
    gamma: f64,
    beta: f64,
) -> Vec<f64> {
    let std_inv = 1.0 / (var + eps).sqrt();
    let mut out = Vec::with_capacity(input.len());
    for &v in input {
        out.push((v - mean) * std_inv * gamma + beta);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(
        unused_imports,
        unused_variables,
        unused_mut,
        dead_code,
        clippy::approx_constant,
        clippy::needless_range_loop,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::too_many_arguments,
        clippy::doc_markdown
    )]
    use super::*;
    use crate::augment::*;
    use crate::config::*;
    use crate::consistency::*;
    use crate::core::*;
    use crate::curriculum::*;
    use crate::decay::*;
    use crate::dropout::*;
    use crate::dropout_uncertainty::*;
    use crate::earlystop::*;
    use crate::label_smooth::*;
    use crate::normalization::*;
    use crate::ops::*;
    use crate::perturb::*;
    use crate::r#impl::*;
    use crate::registry::*;
    use crate::regularizers::*;
    use crate::rules::*;
    use crate::stopping::*;
    use crate::train_hooks::*;
    use crate::utils::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
