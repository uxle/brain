//! # Functional Regularization High-Level Implementations
//!
//! Convenient functional endpoints for applying dropout, batch normalization, layer normalization, and penalties.
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

use super::core::{RegResult, Regularization};
use super::dropout::Dropout;
use super::normalization::layer::{LayerNorm, LayerNormConfig};
use brain_core::Tensor;

/// Functional Dropout application on a single Tensor.
pub fn apply_dropout(tensor: &Tensor, p: f64, is_training: bool) -> RegResult<Tensor> {
    let mut drop = Dropout::new(p);
    if !is_training {
        drop.eval_mode();
    }
    drop.apply(tensor)
}

/// Functional LayerNorm application on a Tensor.
pub fn apply_layernorm(
    tensor: &Tensor,
    normalized_shape: Vec<usize>,
    eps: f64,
) -> RegResult<Tensor> {
    let ln = LayerNorm::new(LayerNormConfig {
        normalized_shape,
        eps,
        elementwise_affine: false,
    });
    ln.forward(tensor)
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
