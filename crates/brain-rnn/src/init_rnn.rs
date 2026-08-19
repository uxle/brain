//! # Recurrent Weight Initializers
//!
//! Orthogonal hidden matrix generation, Xavier input bounds, and forget gate bias presets.
#![allow(missing_docs, clippy::excessive_precision, clippy::approx_constant, clippy::needless_range_loop, clippy::too_many_arguments, clippy::manual_is_multiple_of, clippy::manual_div_ceil, clippy::doc_markdown, clippy::module_inception, clippy::manual_memcpy)]

use brain_core::Tensor;
use super::utils::{init_orthogonal, init_uniform};

/// RNN Initialization strategy configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct RnnInitConfig {
    pub orthogonal_hidden: bool,
    pub forget_bias_init: f64,
    pub seed: u64,
}

impl Default for RnnInitConfig {
    fn default() -> Self {
        Self {
            orthogonal_hidden: true,
            forget_bias_init: 1.0,
            seed: 42,
        }
    }
}

/// Initializes 4-gate LSTM weight tensors according to config.
pub fn init_lstm_weights(input_dim: usize, hidden_dim: usize, config: &RnnInitConfig) -> (Tensor, Tensor, Tensor) {
    let w_ih = init_uniform(4 * hidden_dim, input_dim, input_dim, config.seed);
    let w_hh = if config.orthogonal_hidden {
        init_orthogonal(4 * hidden_dim, hidden_dim, config.seed + 1)
    } else {
        init_uniform(4 * hidden_dim, hidden_dim, hidden_dim, config.seed + 1)
    };

    let mut b_data = vec![0.0; 4 * hidden_dim];
    for i in hidden_dim..(2 * hidden_dim) {
        b_data[i] = config.forget_bias_init;
    }
    let bias = Tensor::from_slice(&b_data, vec![4 * hidden_dim]);

    (w_ih, w_hh, bias)
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables, unused_mut, dead_code, clippy::approx_constant, clippy::needless_range_loop, clippy::manual_div_ceil, clippy::manual_is_multiple_of, clippy::too_many_arguments, clippy::doc_markdown, clippy::excessive_precision)]
    use super::*;
    use crate::core::*;
    use crate::config::*;
    use crate::utils::*;
    use crate::ops::*;
    use crate::cells::*;
    use crate::seq::*;
    use crate::init_rnn::*;
    use crate::reg_ops::*;
    use crate::process::*;
    use crate::backward_ops::*;
    use crate::builder::*;
    use crate::helper::*;
    use crate::VERSION;
    use brain_core::Tensor;
}
